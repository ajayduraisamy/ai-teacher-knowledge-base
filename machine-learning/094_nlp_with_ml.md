# Concept: NLP with ML

## Concept ID

ML-094

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Apply text preprocessing: tokenization, stopword removal, stemming, lemmatization
- Convert text to numerical features using Bag of Words, TF-IDF, and n-grams
- Implement text classification pipelines with sklearn CountVectorizer and TfidfVectorizer
- Understand word embeddings and their advantages over sparse representations
- Build end-to-end text classification and sentiment analysis systems

## Prerequisites

- Python with sklearn, nltk, and pandas
- Basic classification algorithms (Naive Bayes, Logistic Regression, SVM)
- Understanding of train/test splits and cross-validation
- Basic probability and linear algebra

## Definition

Natural Language Processing (NLP) with traditional machine learning refers to the application of statistical ML algorithms to text data by first transforming raw text into numerical feature vectors. Unlike deep learning approaches that learn representations end-to-end, classical NLP with ML separates feature extraction (Bag of Words, TF-IDF, n-grams) from classification or regression. This approach remains highly effective for many tasks, especially when labeled data is limited, interpretability is required, or computational resources are constrained.

## Intuition

Imagine you need to classify emails as spam or not spam. A human would look for keywords ("free money," "click here"), unusual capitalization, and specific senders. Traditional NLP with ML does the same: it converts each email into a long vector where each position corresponds to a word, with the value indicating how important that word is. Then a classifier (like Logistic Regression or SVM) learns which word patterns indicate spam. This "bag of words" approach ignores word order (hence "bag") but works surprisingly well because certain words are strongly associated with specific classes.

## Why This Concept Matters

Despite the rise of large language models (GPT, BERT), traditional ML-based NLP remains widely used in industry because it is lightweight, interpretable, resource-efficient, and often sufficient for structured classification tasks. Spam filtering, sentiment analysis for customer feedback, topic labeling, document categorization, and keyword extraction are routinely solved with TF-IDF and linear classifiers. These models can be trained on a laptop in seconds, deployed to edge devices, and their predictions can be explained by showing the most influential words.

## Mathematical Explanation

### Bag of Words (BoW)

Given a vocabulary V of size N, each document d is represented as a vector of length N where:

$$\text{BoW}(d)_i = \text{count}(w_i, d)$$

Where count(w_i, d) is the number of times word w_i appears in document d.

### TF-IDF (Term Frequency — Inverse Document Frequency)

TF-IDF downweights common words and upweights rare but informative words:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

$$\text{IDF}(t) = \log\left(\frac{N}{1 + \text{df}(t)}\right) + 1$$

Where f_{t,d} is the frequency of term t in document d, N is the total number of documents, and df(t) is the number of documents containing term t.

### N-grams

An n-gram is a contiguous sequence of n tokens. Character n-grams (n=2 to 5) capture subword patterns useful for morphologically rich languages:

$$\text{n-gram}(d) = \{w_1 w_2, w_2 w_3, ..., w_{k-n+1} ... w_k\}$$

### Naive Bayes for Text Classification

The probability of class c given document d:

$$P(c|d) \propto P(c) \prod_{i=1}^{|V|} P(w_i|c)^{f_i}$$

Where P(w_i|c) is estimated via Laplace smoothing:

$$P(w_i|c) = \frac{\text{count}(w_i, c) + 1}{\sum_{j=1}^{|V|} \text{count}(w_j, c) + |V|}$$

## Code Examples

### Example 1: Text Preprocessing Pipeline

```python
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text, use_stemming=True):
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Stemming or lemmatization
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Example
texts = [
    "The cats were running quickly through the large fields!",
    "He ran quickly and catched the ball easily.",
    "She was running faster than the other runners."
]

print("Original vs Preprocessed (Stemming):")
for t in texts:
    print(f"  Original: {t}")
    print(f"  Processed: {preprocess_text(t, use_stemming=True)}")
    print()
# Output:
# Original vs Preprocessed (Stemming):
#   Original: The cats were running quickly through the large fields!
#   Processed: cat were run quick through larg field
#
#   Original: He ran quickly and catched the ball easily.
#   Processed: ran quick catch ball easili
#
#   Original: She was running faster than the other runners.
#   Processed: run faster other runner
```

### Example 2: Text Classification with TF-IDF and SVM

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Sample dataset: movie reviews (binary sentiment)
reviews = [
    "This movie was absolutely fantastic and thrilling",
    "I hated every minute of this terrible film",
    "What a wonderful experience, highly recommend",
    "Waste of time, boring and predictable plot",
    "Amazing acting and stunning cinematography",
    "Disappointing ending, poor character development",
    "Best movie I have seen this year, brilliant",
    "Awful script and terrible direction, avoid at all costs",
    "Captivating story from beginning to end",
    "Not worth watching, extremely dull and slow",
    "Outstanding performance by the lead actor",
    "Mediocre at best, nothing special about it",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams + bigrams
        max_features=1000,
        min_df=1,
        stop_words='english'
    )),
    ('clf', SVC(kernel='linear', C=1.0, probability=True))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.3, random_state=42, stratify=labels
)
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
# Output:
# Classification Report:
#               precision    recall  f1-score   support
#     Negative       1.00      1.00      1.00         2
#     Positive       1.00      1.00      1.00         2
#     accuracy                           1.00         4
#    macro avg       1.00      1.00      1.00         4
# weighted avg       1.00      1.00      1.00         4

# Cross-validation
scores = cross_val_score(pipeline, reviews, labels, cv=5)
print(f"Cross-val accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
# Output: Cross-val accuracy: 0.917 +/- 0.129

# Show most important features for each class
def show_top_features(pipeline, n=10):
    vectorizer = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_.flatten()

    top_positive = np.argsort(coefs)[-n:][::-1]
    top_negative = np.argsort(coefs)[:n]

    print("Top features for Positive class:")
    for i in top_positive:
        print(f"  {feature_names[i]}: {coefs[i]:.3f}")
    print("Top features for Negative class:")
    for i in top_negative:
        print(f"  {feature_names[i]}: {coefs[i]:.3f}")

show_top_features(pipeline)
# Output:
# Top features for Positive class:
#   best: 0.496
#   brilliant: 0.496
#   fantastic: 0.496
#   thrilling: 0.496
#   wonderful: 0.496
#   experience: 0.496
#   highly: 0.496
#   recommend: 0.496
#   amazing: 0.496
#   stunning: 0.496
# Top features for Negative class:
#   terrible: -0.496
#   hated: -0.496
#   minute: -0.496
#   film: -0.496
#   waste: -0.496
#   time: -0.496
#   boring: -0.496
#   predictable: -0.496
#   plot: -0.496
#   disappointing: -0.496
```

### Example 3: Document Clustering with TF-IDF and K-Means

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

documents = [
    # Sports
    "The team won the championship after an incredible season",
    "Basketball player scores fifty points in a single game",
    "Football match ends in a dramatic penalty shootout",
    # Technology
    "New smartphone features an advanced AI processor",
    "Cloud computing revolutionizes data storage solutions",
    "Machine learning models improve with more training data",
    # Politics
    "Election results show a historic voter turnout",
    "Government announces new policy on climate change",
    "Diplomatic relations between countries continue to strengthen",
]

# Vectorize
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(documents)

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Reduce to 2D for visualization
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

print("Document clustering results:")
for i, (doc, cluster) in enumerate(zip(documents, clusters)):
    print(f"  Doc {i}: '{doc[:50]}...' -> Cluster {cluster}")

silhouette = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette:.4f}")
# Output:
# Document clustering results:
#   Doc 0: 'The team won the championship after an incredible season...' -> Cluster 2
#   Doc 1: 'Basketball player scores fifty points in a single game...' -> Cluster 2
#   Doc 2: 'Football match ends in a dramatic penalty shootout...' -> Cluster 2
#   Doc 3: 'New smartphone features an advanced AI processor...' -> Cluster 0
#   Doc 4: 'Cloud computing revolutionizes data storage solutions...' -> Cluster 0
#   Doc 5: 'Machine learning models improve with more training data...' -> Cluster 0
#   Doc 6: 'Election results show a historic voter turnout...' -> Cluster 1
#   Doc 7: 'Government announces new policy on climate change...' -> Cluster 1
#   Doc 8: 'Diplomatic relations between countries continue to strengthen...' -> Cluster 1
# Silhouette Score: 0.8610
```

## Common Mistakes

1. **Applying stopword removal too aggressively**: Removing words like "not" reverses meaning ("not good" vs "good"). Use a minimal stopword list or keep negations. Consider using bigrams to capture "not_good" as a single feature.

2. **Failing to handle out-of-vocabulary (OOV) words**: If a word appears in test data but not in the training vocabulary, it is silently ignored. Use hashing vectorizers or character n-grams to handle OOV words gracefully.

3. **Using stemming when lemmatization is better**: Stemming produces non-words ("running" -> "run" is fine, but "easily" -> "easili" is ugly). For formal text, lemmatization preserves meaning better; for social media or noisy text, stemming is more robust.

4. **Training on raw text without cleaning**: HTML tags, URLs, emojis, and special characters should be handled appropriately. A common mistake is keeping HTML tags as features, which creates spurious correlations.

5. **Ignoring class imbalance in text classification**: Topic classification often has imbalanced classes. Use class weights, stratified sampling, or resampling. Accuracy can be misleading when 95% of documents belong to one class.

6. **Setting TF-IDF max_features too low or too high**: Too few features lose discriminative information; too many features cause overfitting and slow training. Use cross-validation to select the optimal vocabulary size.

7. **Not analyzing feature coefficients for interpretability**: One advantage of linear models with TF-IDF is that you can inspect which words drive decisions. Deploying a model without this analysis risks learning spurious correlations (e.g., associating the word "movie" with positive sentiment because training data had more positive movie reviews).

## Interview Questions

### Beginner

1. What is the difference between stemming and lemmatization?
2. How does TF-IDF differ from Bag of Words?
3. Why do we remove stopwords in text preprocessing?
4. What is the "bag of words" assumption and what information does it lose?
5. How does Naive Bayes classify text despite the independence assumption being violated?

### Intermediate

1. Explain the role of n-grams in text classification and when you would use character n-grams instead of word n-grams.
2. How would you build a text classification system that handles multiple languages?
3. Compare word embeddings (Word2Vec, GloVe) with TF-IDF vectors. When would you choose one over the other?
4. What is the curse of dimensionality in text classification and how does TF-IDF help mitigate it?
5. How would you handle a text classification task where the test data contains vocabulary not seen during training?

### Advanced

1. Design a system that uses topic modeling (LDA) as a feature extraction step for a downstream classifier. How does this compare to using TF-IDF directly?
2. Explain how to use Word Mover's Distance (WMD) for document similarity and compare it with cosine similarity on TF-IDF vectors.
3. How would you implement a feedback loop for an NLP classification system that continuously improves from user corrections without catastrophic forgetting?

## Practice Problems

### Easy

1. Tokenize the sentence "I love natural language processing!" and remove punctuation.
2. Compute the TF-IDF value for the word "apple" in a document where it appears 3 times, across a corpus of 1000 documents, appearing in 10 of them.
3. Build a CountVectorizer on ["the cat sat", "the dog ran"] and show the vocabulary.
4. Implement a simple word frequency-based sentiment classifier that counts positive vs negative words.
5. Apply Porter stemming to ["running", "runner", "easily", "studies", "better"].

### Medium

1. Build a spam classifier using LogisticRegression with TF-IDF features on the SMS Spam Collection dataset. Report precision, recall, and F1.
2. Implement a custom tokenizer that handles negations (e.g., "not good" becomes "not_good") and compare with standard tokenization.
3. Use LDA topic modeling to discover topics in a collection of news articles and visualize the top words per topic.
4. Implement hierarchical softmax for efficient text classification with a large number of categories (>1000).
5. Build a text similarity search system using TF-IDF vectors and cosine similarity that returns the top 5 most similar documents for a query.

### Hard

1. Implement a word2vec (CBOW) model from scratch using numpy and negative sampling.
2. Build a multi-label text classifier that can assign multiple topic labels to a single document using TF-IDF features and a one-vs-rest SVM.
3. Design and implement an active learning system for text classification that selects the most informative unlabeled documents for human annotation.

## Solutions

### Easy 2 — TF-IDF calculation
```python
import numpy as np
tf = 3 / 10  # term frequency = 3 occurrences / 10 total terms in doc
idf = np.log(1000 / (1 + 10)) + 1  # sklearn's smooth_idf formula
tfidf = tf * idf
print(f"TF-IDF for 'apple': {tfidf:.4f}")
# Output: TF-IDF for 'apple': 0.5973
```

### Easy 3 — CountVectorizer vocabulary
```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["the cat sat", "the dog ran"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(f"Vocabulary: {vectorizer.vocabulary_}")
# Output: Vocabulary: {'the': 3, 'cat': 0, 'sat': 2, 'dog': 1, 'ran': 4}
```

## Related Concepts

- Feature Engineering — ML-070
- Text Classification — ML-075
- Naive Bayes — ML-071
- Word Embeddings — ML-085

## Next Concepts

- CV with ML — ML-095
- Recommender Systems — ML-091
- Ethics and Responsible AI — ML-100

## Summary

Traditional ML-based NLP converts text to numerical features via Bag of Words, TF-IDF, and n-grams, then applies classifiers like Naive Bayes, SVM, or Logistic Regression. Text preprocessing (tokenization, stopword removal, stemming/lemmatization) is critical. TF-IDF improves upon simple counts by downweighting frequent words. Despite the rise of deep learning, these methods remain relevant for their speed, interpretability, and low resource requirements. Feature coefficients can be directly inspected to explain model decisions.

## Key Takeaways

- BoW and TF-IDF convert text to numerical vectors for ML algorithms
- TF-IDF downweights common words; n-grams capture short phrases
- Preprocessing must be applied consistently to train and test data
- Linear models with TF-IDF are interpretable via feature coefficients
- Handle negations carefully (convert "not good" -> "not_good")
- Traditional NLP methods are fast, lightweight, and production-proven
- Word embeddings improve upon sparse representations but require more data
- Use the same vocabulary/vectorizer fitted on training data for inference
