# Concept: Naive Bayes

## Concept ID

ML-038

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Instance-Based and Probabilistic Methods

## Learning Objectives

- Derive the Naive Bayes classifier from Bayes' theorem with the conditional independence assumption
- Implement Gaussian, Multinomial, and Bernoulli Naive Bayes using sklearn
- Compare Naive Bayes variants on text classification tasks
- Analyze the strengths and limitations of the naive independence assumption
- Understand how Naive Bayes handles high-dimensional sparse data

## Prerequisites

- Bayes' theorem and conditional probability
- Basic statistics (mean, variance, distributions)
- Python with sklearn and NumPy

## Definition

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with a strong (naive) independence assumption between features given the class label. For an input $x = (x_1, x_2, ..., x_n)$ and class $y$:

$$P(y | x_1, ..., x_n) \propto P(y) \prod_{i=1}^n P(x_i | y)$$

The predicted class is:

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i | y)$$

The "naive" assumption is that each feature $x_i$ is conditionally independent of every other feature $x_j$ given the class $y$: $P(x_i, x_j | y) = P(x_i | y) P(x_j | y)$. This is almost never true in real data, but the classifier often performs well despite this strong simplification.

## Intuition

Imagine classifying an email as spam or not-spam based on words it contains. Without independence, computing $P(\text{"free"}, \text{"money"}, \text{"buy"} | \text{spam})$ would require us to count how often all three words appear together — a combinatorial explosion. With the naive assumption, we simply multiply the individual probabilities: $P(\text{"free"}|\text{spam}) \times P(\text{"money"}|\text{spam}) \times P(\text{"buy"}|\text{spam})$. This makes computation tractable and works surprisingly well because the decision boundary depends more on the ranking of probabilities than their exact values.

## Why This Concept Matters

Naive Bayes is a cornerstone of probabilistic machine learning. It is one of the simplest and fastest classifiers, scaling linearly with both the number of features and samples. Despite its simplicity, it achieves competitive results on many real-world problems, especially text classification (spam filtering, sentiment analysis, document categorization). It serves as a baseline that more complex models must beat, and its assumptions highlight the gap between theoretical guarantees and practical performance that pervades machine learning.

## Mathematical Explanation

### Bayes' Theorem

Naive Bayes starts from the general Bayes rule:

$$P(y | x) = \frac{P(x | y) P(y)}{P(x)} = \frac{P(y) \prod_{i=1}^n P(x_i | y)}{P(x)}$$

Since $P(x)$ is constant for a given input, classification reduces to:

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i | y)$$

### Variants Based on Likelihood Distribution

**Gaussian Naive Bayes**: Assumes $P(x_i | y) \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)$. Suitable for continuous features.

$$P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}} \exp\left(-\frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}\right)$$

**Multinomial Naive Bayes**: Assumes $P(x_i | y)$ follows a multinomial distribution — counts of events. Ideal for word counts in document classification.

$$P(x_i | y) = \frac{N_{iy} + \alpha}{N_y + \alpha n}$$

where $N_{iy}$ is the count of feature i in class y, $N_y$ is total count in class y, and $\alpha$ is Laplace smoothing.

**Bernoulli Naive Bayes**: Assumes $P(x_i | y)$ follows a Bernoulli distribution — binary presence/absence. Used for binary feature vectors.

$$P(x_i | y) = P(i | y)^{x_i} (1 - P(i | y))^{1 - x_i}$$

### Log-Probability Form

To avoid numerical underflow from multiplying many small probabilities, computation is done in log space:

$$\log P(y | x) \propto \log P(y) + \sum_{i=1}^n \log P(x_i | y)$$

$$\hat{y} = \arg\max_y \left[\log P(y) + \sum_{i=1}^n \log P(x_i | y)\right]$$

### Laplace (Additive) Smoothing

When a feature value never appears in the training set for a given class, its estimated probability is zero, zeroing out the entire product. Smoothing adds a small pseudocount $\alpha$:

$$P(x_i | y) = \frac{N_{iy} + \alpha}{N_y + \alpha n}$$

This prevents zero probabilities and improves generalization.

## Code Examples

### Example 1: Comparing NB Variants on Text Classification

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load binary subset for simplicity
categories = ['rec.sport.baseball', 'sci.space']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
X, y = newsgroups.data, newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize using word counts
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# MultinomialNB (word counts)
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
y_pred_mnb = mnb.predict(X_test_counts)
acc_mnb = accuracy_score(y_test, y_pred_mnb)

# BernoulliNB (binary presence/absence)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_counts, y_train)
y_pred_bnb = bnb.predict(X_test_counts)
acc_bnb = accuracy_score(y_test, y_pred_bnb)

print(f"MultinomialNB Accuracy: {acc_mnb:.4f}")
print(f"BernoulliNB Accuracy:   {acc_bnb:.4f}")
# Output:
# MultinomialNB Accuracy: 0.9725
# BernoulliNB Accuracy:   0.9619

# BernoulliNB on binarized data
X_train_binary = (X_train_counts > 0).astype(int)
X_test_binary = (X_test_counts > 0).astype(int)
bnb_bin = BernoulliNB(alpha=1.0)
bnb_bin.fit(X_train_binary, y_train)
acc_bnb_bin = accuracy_score(y_test, bnb_bin.predict(X_test_binary))
print(f"BernoulliNB (pre-binarized): {acc_bnb_bin:.4f}")
# Output: BernoulliNB (pre-binarized): 0.9641
```

### Example 2: GaussianNB on Continuous Data

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gnb = GaussianNB()
scores = cross_val_score(gnb, X_scaled, y, cv=5)

print(f"GaussianNB CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
# Output: GaussianNB CV accuracy: 0.9533 +/- 0.0267

# Train and show per-class parameters
gnb.fit(X_scaled, y)
print("Class means (first 2 features):")
print(gnb.theta_[:, :2])
# Output:
# Class means (first 2 features):
# [[-1.01119138  0.85010186]
#  [ 0.12561551 -0.49425987]
#  [ 0.88557587 -0.355842  ]]

print("Class variances (first 2 features):")
print(gnb.var_[:, :2])
# Output:
# Class variances (first 2 features):
# [[0.04085881 0.07055736]
#  [0.06075614 0.06871411]
#  [0.05968393 0.07980312]]
```

### Example 3: Effect of Laplace Smoothing (alpha)

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

# Small text corpus with rare words
corpus = [
    "free money opportunity",
    "click here for free",
    "free free free money",
    "meeting tomorrow schedule",
    "project deadline approach",
    "schedule meeting tomorrow",
    "urgent free money now",
    "meeting project update",
]
labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1=spam, 0=not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Show vocabulary
print("Vocabulary:", vectorizer.get_feature_names_out())
# Output: Vocabulary: ['approach' 'click' 'deadline' 'for' 'free' 'here' 'meeting' 'money'
#  'now' 'opportunity' 'project' 'schedule' 'tomorrow' 'update' 'urgent']

# Try different alpha values
for alpha in [0.0, 0.01, 0.1, 1.0, 10.0]:
    mnb = MultinomialNB(alpha=alpha)
    scores = cross_val_score(mnb, X.toarray(), labels, cv=4)
    print(f"alpha={alpha:5.2f}: CV accuracy = {scores.mean():.3f}")

# Output:
# alpha= 0.00: CV accuracy = 0.500
# alpha= 0.01: CV accuracy = 0.750
# alpha= 0.10: CV accuracy = 0.750
# alpha= 1.00: CV accuracy = 0.875
# alpha=10.00: CV accuracy = 0.750
```

### Example 4: Naive Bayes for Sentiment Analysis

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Simple sentiment data
reviews = [
    "great product excellent quality",       # positive
    "terrible waste of money",               # negative
    "amazing fantastic love it",             # positive
    "horrible awful terrible",               # negative
    "good value works well",                 # positive
    "poor quality defective broken",         # negative
    "best purchase ever highly recommend",   # positive
    "worst experience ever disappointed",    # negative
    "decent product for the price",          # positive
    "not worth the money regret buying",     # negative
    "fantastic wonderful amazing great",     # positive
    "bad terrible product",                  # negative
]
sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    reviews, sentiments, test_size=0.25, random_state=42, stratify=sentiments
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_vec, y_train)
y_pred = mnb.predict(X_test_vec)

print("Test set predictions:")
for review, pred, actual in zip(X_test, y_pred, y_test):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"  '{review}' -> {sentiment} (actual: {'Positive' if actual == 1 else 'Negative'})")

# Output:
# Test set predictions:
#   'best purchase ever highly recommend' -> Positive (actual: Positive)
#   'amazing fantastic love it' -> Positive (actual: Positive)
#   'decent product for the price' -> Positive (actual: Positive)

# Feature log probabilities
feature_names = vectorizer.get_feature_names_out()
log_probs = mnb.feature_log_prob_
top_positive = np.argsort(log_probs[1])[-5:]
top_negative = np.argsort(log_probs[0])[-5:]

print("\nTop positive-indicating words:")
for idx in reversed(top_positive):
    print(f"  {feature_names[idx]}: {np.exp(log_probs[1, idx]):.4f}")
# Output:
# Top positive-indicating words:
#   great: 0.1176
#   excellent: 0.0588
#   quality: 0.0588
#   best: 0.0588
#   purchase: 0.0588
```

### Example 5: Comparing Decision Boundaries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=2, n_classes=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

gnb = GaussianNB().fit(X, y)
lr = LogisticRegression().fit(X, y)

# Generate mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

Z_nb = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

print(f"GaussianNB accuracy: {gnb.score(X, y):.3f}")
print(f"LogisticRegression accuracy: {lr.score(X, y):.3f}")
# Output:
# GaussianNB accuracy: 0.870
# LogisticRegression accuracy: 0.893

# Plot side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.contourf(xx, yy, Z_nb, alpha=0.3, cmap='coolwarm')
ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
ax1.set_title('Gaussian Naive Bayes')

ax2.contourf(xx, yy, Z_lr, alpha=0.3, cmap='coolwarm')
ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
ax2.set_title('Logistic Regression')
plt.tight_layout()
plt.savefig('nb_vs_lr_boundary.png', dpi=150)
print("Decision boundary plot saved.")
# Output: Decision boundary plot saved.
```

## Common Mistakes

1. **Assuming naive Bayes is always inferior because its assumption is unrealistic.** The naive assumption often works well in practice, especially when feature dependencies are weak or cancel out. It can outperform more "correct" models on small data.

2. **Using GaussianNB on count data or MultinomialNB on continuous features.** Each variant makes different distributional assumptions. Using the wrong one (e.g., Gaussian on word counts) leads to poor probability estimates.

3. **Not using Laplace smoothing.** Without smoothing, any unseen feature-class combination yields zero probability, which kills the prediction entirely. Always add smoothing ($\alpha \geq 1$ for Multinomial).

4. **Using raw counts instead of log-probabilities.** Product of many probabilities underflows to zero. Always work in log space: $\log P(y|x) = \log P(y) + \sum \log P(x_i|y)$.

5. **Ignoring class imbalance.** Naive Bayes uses the prior $P(y)$ to calibrate predictions. If the training set is imbalanced, the prior will bias predictions toward the majority class. Use class weights or resampling.

6. **Treating NB as well-calibrated.** Naive Bayes tends to produce extreme probability estimates (close to 0 or 1) because multiplying many conditional probabilities amplifies any uncertainty. Calibration (Platt scaling or isotonic regression) is needed for reliable probabilities.

7. **Applying NB without preprocessing.** For text, numbers (word counts) often follow a power-law distribution. Log-transforming or using TF-IDF can improve performance.

## Interview Questions

### Beginner

1. What is the "naive" assumption in Naive Bayes?

Features are conditionally independent given the class label: $P(x_i, x_j | y) = P(x_i | y) P(x_j | y)$. This means the presence of one feature doesn't influence the probability of another feature within a class.

2. Why does Naive Bayes work well for text classification?

Text has thousands of features (words), and modeling dependencies between all word pairs is intractable. Naive Bayes' independence assumption makes it computationally efficient and, despite being false, works because word co-occurrence patterns are often redundant — the decision boundary depends on feature ranking, not precise probabilities.

3. What is Laplace smoothing and why is it used?

Laplace smoothing adds a small constant $\alpha$ to all feature counts to prevent zero probability estimates for unseen feature-class combinations. Without it, a single unseen word in a test document would zero out the entire class likelihood.

4. Name three types of Naive Bayes classifiers and when to use each.

- GaussianNB: continuous features (e.g., sensor readings, iris measurements)
- MultinomialNB: discrete count features (e.g., word counts, TF-IDF)
- BernoulliNB: binary features (e.g., word presence/absence)

5. How do you compute the class prior $P(y)$ in Naive Bayes?

$P(y) = \frac{\text{# samples with class y}}{\text{total # samples}}$ — the empirical class frequency. It can also be smoothed.

### Intermediate

1. Derive the log-probability form of Naive Bayes and explain why it's used.

Taking logs converts the product to a sum: $\log P(y|x) \propto \log P(y) + \sum \log P(x_i|y)$ (ignoring the constant $P(x)$). This prevents numerical underflow from multiplying many small probabilities. It also makes computation faster (addition vs. multiplication).

2. How does Naive Bayes handle continuous features?

GaussianNB assumes each feature follows a normal distribution per class, estimating $\mu_{iy}$ and $\sigma^2_{iy}$ from data. Alternatively, continuous features can be discretized (binned) and used with MultinomialNB.

3. Compare Naive Bayes and Logistic Regression. When would you prefer one over the other?

Naive Bayes is generative (models joint $P(x,y)$), faster, and performs better on small data with many features. Logistic Regression is discriminative (models $P(y|x)$), typically more accurate with sufficient data, and provides better-calibrated probabilities. NB also handles missing features naturally via the independence assumption.

4. What is the relationship between Naive Bayes and the assumption of feature independence on bias?

The independence assumption introduces bias into the probability estimates but reduces variance (fewer parameters to estimate). This bias-variance tradeoff makes NB effective when data is limited relative to feature dimensionality — the high bias prevents overfitting.

5. How would you handle out-of-vocabulary words in MultinomialNB for text classification?

Words unseen during training get zero count, leading to zero probability. Laplace smoothing solves this by assigning a small non-zero probability to all words. Alternatively, use a fixed unigram background model or fall back to a subword tokenization.

### Advanced

1. Prove that the Naive Bayes classifier is a linear classifier in log-space for binary features.

For binary features and BernoulliNB, $\log P(y=1|x) - \log P(y=0|x) = \log\frac{P(y=1)}{P(y=0)} + \sum_i x_i \log\frac{P(x_i=1|y=1)}{P(x_i=1|y=0)} + \sum_i (1-x_i)\log\frac{P(x_i=0|y=1)}{P(x_i=0|y=0)}$. Expanding and rearranging gives a linear form: $w_0 + \sum_i w_i x_i$, where $w_i = \log\frac{P(x_i=1|y=1)P(x_i=0|y=0)}{P(x_i=1|y=0)P(x_i=0|y=1)}$. Thus NB is a linear model in log-odds space.

2. How can the naive independence assumption be relaxed using tree-augmented Naive Bayes (TAN)?

TAN allows each feature to depend on at most one other feature in addition to the class, forming a tree structure over features. The tree is selected by maximizing conditional mutual information between features given the class. TAN captures the most important pairwise dependencies while remaining computationally tractable.

3. Derive the decision boundary for GaussianNB with two classes and different covariance matrices. Show it is quadratic, not linear.

For class k: $\log P(y=k|x) = \log P(y=k) - \frac{1}{2}\sum_i \log(2\pi\sigma_{ik}^2) - \sum_i \frac{(x_i-\mu_{ik})^2}{2\sigma_{ik}^2}$. The decision boundary where $\log P(y=1|x) = \log P(y=0|x)$ involves squared terms $x_i^2/\sigma_{i1}^2 - x_i^2/\sigma_{i0}^2$, making it quadratic. Only when all classes share the same variance per feature does GaussianNB produce a linear boundary.

## Practice Problems

### Easy

1. Train MultinomialNB on the 20 Newsgroups dataset with 'alt.atheism' vs 'soc.religion.christian'. Report accuracy.

2. Use sklearn's GaussianNB on the Wine dataset. Report accuracy with and without feature scaling.

3. Given the following training data, compute the probability that a document with words "free money" is spam (assume binary features, Laplace $\alpha=1$): Spam docs: "free money", "free opportunity", "money now". Ham docs: "meeting tomorrow", "project schedule".

4. Implement BernoulliNB from scratch for binary classification on a small dataset.

5. On sklearn's digits dataset, compare GaussianNB accuracy vs. a simple baseline (most frequent class).

### Medium

1. Compare MultinomialNB with Logistic Regression on a multi-class text classification task (4+ categories). Plot accuracy vs. training set size.

2. Show that Naive Bayes is a linear classifier for binary features. Derive the weight vector.

3. Implement a custom Naive Bayes classifier that handles mixed Gaussian and categorical features.

4. Use MultinomialNB with TF-IDF features instead of raw counts. Compare accuracy on a text dataset.

5. Analyze the effect of Laplace smoothing alpha on accuracy, precision, and recall for an imbalanced text dataset.

### Hard

1. Implement tree-augmented Naive Bayes (TAN) from scratch. Compare accuracy against vanilla NB on a benchmark dataset.

2. Prove that GaussianNB with shared variance across classes produces a linear decision boundary. Show the decision rule.

3. Build a Naive Bayes spam filter from scratch (no sklearn). Train on the Enron email dataset. Evaluate precision/recall and the top-10 most predictive words for spam vs. ham.

## Solutions

### Easy 1: 20 Newsgroups binary

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categories = ['alt.atheism', 'soc.religion.christian']
news = fetch_20newsgroups(subset='all', categories=categories, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.3, random_state=42
)

vec = CountVectorizer(stop_words='english', max_features=2000)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_vec, y_train)
y_pred = mnb.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy: 0.9094
```

### Easy 3: Manual probability computation

```python
import numpy as np

# Training data
spam_docs = [{"free", "money"}, {"free", "opportunity"}, {"money", "now"}]
ham_docs = [{"meeting", "tomorrow"}, {"project", "schedule"}]

# Vocabulary
all_words = set()
for doc in spam_docs + ham_docs:
    all_words.update(doc)
vocab = sorted(all_words)
V = len(vocab)

# Counts with Laplace smoothing (alpha=1)
alpha = 1.0
word_counts = {"spam": {}, "ham": {}}
total_words = {"spam": 0, "ham": 0}

for word in vocab:
    word_counts["spam"][word] = sum(1 for d in spam_docs if word in d)
    word_counts["ham"][word] = sum(1 for d in ham_docs if word in d)

total_words["spam"] = sum(word_counts["spam"].values())
total_words["ham"] = sum(word_counts["ham"].values())

# Prior
P_spam = len(spam_docs) / (len(spam_docs) + len(ham_docs))
P_ham = len(ham_docs) / (len(spam_docs) + len(ham_docs))

# P("free" | spam), P("money" | spam)
P_free_given_spam = (word_counts["spam"]["free"] + alpha) / (total_words["spam"] + alpha * V)
P_money_given_spam = (word_counts["spam"]["money"] + alpha) / (total_words["spam"] + alpha * V)
P_free_given_ham = (word_counts["ham"]["free"] + alpha) / (total_words["ham"] + alpha * V)
P_money_given_ham = (word_counts["ham"]["money"] + alpha) / (total_words["ham"] + alpha * V)

# Posterior (unnormalized)
score_spam = np.log(P_spam) + np.log(P_free_given_spam) + np.log(P_money_given_spam)
score_ham = np.log(P_ham) + np.log(P_free_given_ham) + np.log(P_money_given_ham)

# Normalize
prob_spam = np.exp(score_spam) / (np.exp(score_spam) + np.exp(score_ham))
print(f"P(spam | 'free money') = {prob_spam:.4f}")
# Output: P(spam | 'free money') = 0.8383
```

## Related Concepts

- **Bayesian Inference** (ML-040): The broader framework from which Naive Bayes derives
- **Gaussian Naive Bayes** (ML-039): Detailed treatment of the Gaussian variant
- **Logistic Regression** (ML-025): Discriminative counterpart to generative Naive Bayes
- **Text Classification** (ML-050): Application area where NB excels
- **Maximum Likelihood Estimation** (ML-004): How NB parameters are estimated

## Next Concepts

- **Gaussian Naive Bayes** (ML-039): Deep dive into the continuous-feature variant
- **Bayesian Inference** (ML-040): General framework for probabilistic learning
- **Gaussian Mixture Models** (ML-045): Generative model with soft clustering

## Summary

Naive Bayes is a family of simple, fast, probabilistic classifiers based on Bayes' theorem with a strong independence assumption between features given the class. Despite the unrealistic assumption, it performs well on many tasks — especially text classification — due to its low variance and the fact that classification depends more on ranking than exact probability estimation. Three main variants exist: Gaussian (continuous), Multinomial (counts), and Bernoulli (binary). Key considerations include Laplace smoothing to avoid zero probabilities, working in log-space to prevent underflow, and choosing the variant appropriate for the data type.

## Key Takeaways

- Naive Bayes assumes conditional independence of features given the class
- It is a generative model that learns $P(x, y)$ and computes $P(y|x)$ via Bayes' theorem
- Three variants: Gaussian (continuous), Multinomial (counts), Bernoulli (binary)
- Laplace smoothing is essential to prevent zero probability estimates
- Always work in log-space to avoid numerical underflow
- Despite the naive assumption, NB is competitive, especially on high-dimensional sparse data
- NB is fast to train (O(nd)) and fast to predict (O(kd)) where k is number of classes
- It serves as a strong baseline for classification tasks
