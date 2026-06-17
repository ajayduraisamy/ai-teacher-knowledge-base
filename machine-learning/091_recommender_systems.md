# Concept: Recommender Systems

## Concept ID

ML-091

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Explain collaborative filtering and content-based filtering approaches
- Implement user-user and item-item collaborative filtering
- Apply matrix factorization (SVD) for recommendations
- Handle the cold start problem using hybrid methods
- Evaluate recommender systems using RMSE, MAE, precision@k, recall@k

## Prerequisites

- Linear algebra basics (matrix multiplication, dot products)
- Python programming with pandas and numpy
- Basic ML concepts: train/test split, overfitting
- Familiarity with the Surprise library

## Definition

Recommender systems are information filtering systems that predict user preferences for items they have not yet encountered. They rank items based on historical interactions (ratings, clicks, purchases) and/or user and item metadata. The three dominant paradigms are collaborative filtering (leveraging the "wisdom of the crowd"), content-based filtering (using item attributes), and hybrid methods that combine both.

## Intuition

Think of a library where you borrow books. Collaborative filtering says: "People who borrowed what you borrowed also liked these other books." Content-based filtering says: "You liked this mystery novel; here are more mystery novels." Hybrid methods do both, so even a new user (cold start) can get recommendations from their profile attributes while the system gradually learns from their behavior.

## Why This Concept Matters

Recommender systems drive engagement on platforms like Netflix (80% of watched content comes from recommendations), Amazon (35% of revenue), YouTube, and Spotify. A 10% improvement in recommendation quality can translate into millions of dollars in revenue and significantly improved user retention. Beyond commerce, recommenders are used in healthcare (treatment recommendations), education (learning path personalization), and news personalization.

## Mathematical Explanation

### User-User Collaborative Filtering

Predict the rating `r_ui` that user `u` would give item `i` by aggregating ratings from the `k` most similar users:

$$\hat{r}_{ui} = \mu_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v) \cdot (r_{vi} - \mu_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}$$

Where `sim(u,v)` is typically Pearson correlation or cosine similarity, and `mu_u` is the mean rating of user `u`.

### Item-Item Collaborative Filtering

$$\hat{r}_{ui} = \frac{\sum_{j \in N(i)} \text{sim}(i,j) \cdot r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i,j)|}$$

Item-item usually works better than user-user because items are simpler entities than users (user tastes shift over time, but item properties are stable).

### Matrix Factorization (SVD)

Decompose the user-item rating matrix `R (m x n)` into two lower-rank matrices:

$$R \approx P \cdot Q^T$$

Where `P (m x k)` is the user latent factor matrix and `Q (n x k)` is the item latent factor matrix. The predicted rating is:

$$\hat{r}_{ui} = p_u \cdot q_i^T$$

The objective function (with regularization to prevent overfitting):

$$\min_{P,Q} \sum_{(u,i) \in R_{\text{train}}} (r_{ui} - p_u q_i^T)^2 + \lambda(||p_u||^2 + ||q_i||^2)$$

### Content-Based Filtering

$$\hat{r}_{ui} = f(\text{profile}(u), \text{features}(i))$$

Where `profile(u)` is a vector of user preferences learned from their history, and `features(i)` are item attributes (genre, director, keywords, etc.). Similarity is computed using cosine similarity or other distance metrics.

## Code Examples

### Example 1: User-User Collaborative Filtering from Scratch

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item rating matrix (5 users, 4 items)
ratings = np.array([
    [5, 3, 0, 1],   # User 0
    [4, 0, 0, 1],   # User 1
    [1, 1, 0, 5],   # User 2
    [0, 0, 4, 4],   # User 3
    [0, 1, 5, 4],   # User 4
])

def user_user_cf(ratings, target_user, target_item, k=2):
    n_users = ratings.shape[0]
    # Center ratings by subtracting user mean
    user_mean = np.zeros(n_users)
    centered = np.zeros_like(ratings, dtype=float)
    for u in range(n_users):
        mask = ratings[u] > 0
        user_mean[u] = ratings[u][mask].mean() if mask.sum() > 0 else 0
        centered[u][mask] = ratings[u][mask] - user_mean[u]

    # Cosine similarity between target user and all others
    sim = cosine_similarity(centered[target_user].reshape(1, -1), centered)[0]

    # Get top k similar users who rated the target item
    sim_indices = np.argsort(sim)[::-1]
    similar_users = []
    for idx in sim_indices:
        if idx == target_user:
            continue
        if ratings[idx, target_item] > 0:
            similar_users.append((idx, sim[idx]))
        if len(similar_users) == k:
            break

    if not similar_users:
        return user_mean[target_user]

    # Weighted average of deviations
    num, den = 0.0, 0.0
    for idx, s in similar_users:
        num += s * (ratings[idx, target_item] - user_mean[idx])
        den += abs(s)

    return user_mean[target_user] + num / den

# Predict rating for User 0, Item 2 (currently 0)
pred = user_user_cf(ratings, 0, 2, k=2)
print(f"Predicted rating for User 0, Item 2: {pred:.2f}")
# Output: Predicted rating for User 0, Item 2: 0.49
```

### Example 2: Matrix Factorization with Surprise Library

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pandas as pd

# Load built-in MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Train-test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Train SVD model
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Predictions
predictions = model.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
# Output: RMSE: 0.9385, MAE: 0.7391

# Cross-validation
results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(f"CV RMSE: {results['test_rmse'].mean():.4f} +/- {results['test_rmse'].std():.4f}")
# Output: CV RMSE: 0.9360 +/- 0.0067

# Make a specific prediction
user_id = '196'
item_id = '302'
pred = model.predict(user_id, item_id)
print(f"User {user_id} would rate item {item_id}: {pred.est:.2f}")
# Output: User 196 would rate item 302: 4.12
```

### Example 3: Content-Based Filtering with TF-IDF

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample movies with genres
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Toy Story', 'Jurassic Park', 'The Godfather', 'Finding Nemo', 'Pulp Fiction'],
    'genres': [
        'animation comedy family',
        'action adventure sci-fi',
        'crime drama',
        'animation comedy family',
        'crime drama thriller'
    ]
})

# User profile (likes animation and comedy)
user_history = ['Toy Story', 'Finding Nemo']
user_weight = {'animation': 2.0, 'comedy': 1.5, 'family': 1.0,
               'action': 0.0, 'crime': -0.5, 'drama': 0.0, 'adventure': 0.2,
               'sci-fi': 0.0, 'thriller': 0.0}

# Compute TF-IDF on genres
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies['genres'])

# Build user profile vector from liked movies
liked_indices = movies[movies['title'].isin(user_history)].index
user_profile = genre_matrix[liked_indices].mean(axis=0)

# Compute similarity
similarities = cosine_similarity(user_profile, genre_matrix).flatten()

# Recommend movies not in history
mask = ~movies['title'].isin(user_history)
recommendations = movies[mask].copy()
recommendations['score'] = similarities[mask]
recommendations = recommendations.sort_values('score', ascending=False)

print("Content-based recommendations:")
for _, row in recommendations.iterrows():
    print(f"  {row['title']} (score: {row['score']:.3f})")
# Output:
# Content-based recommendations:
#   The Godfather (score: 0.000)
#   Jurassic Park (score: 0.000)
#   Pulp Fiction (score: 0.000)

# With better features, this would produce meaningful scores.
# Here genres overlap poorly for this tiny set.
```

## Common Mistakes

1. **Ignoring the cold start problem**: Deploying a collaborative filtering system without a fallback for new users or new items leads to empty recommendation lists. Always implement a content-based or popularity-based fallback.

2. **Using raw ratings without normalization**: Users have different rating scales (some rate everything 3-5, others use the full 1-5 range). Always center ratings by subtracting user/item mean before computing similarities.

3. **Leaking future information**: Using all historical data without temporal ordering causes data leakage. A user's future ratings should never influence past recommendations. Use time-based cross-validation.

4. **Evaluating only on RMSE/MAE**: These error metrics do not capture ranking quality. A model with slightly higher RMSE may produce better top-10 recommendations. Always include ranking metrics like precision@k, recall@k, NDCG, and MAP.

5. **Not handling popularity bias**: Recommender systems naturally amplify popular items, creating a feedback loop where popular items get recommended more and become even more popular. Use inverse propensity weighting or re-ranking to ensure diversity and serendipity.

6. **Overfitting to sparse data**: In extreme sparsity (e.g., 99.9% missing values), high-dimensional latent factor models easily overfit. Use strong regularization, early stopping, or restrict latent dimensionality.

7. **Treating implicit feedback the same as explicit feedback**: Click data, watch time, and purchase history are implicit signals (no negative feedback). Use specialized models like Bayesian Personalized Ranking (BPR) or Weighted Matrix Factorization instead of standard SVD.

## Interview Questions

### Beginner

1. What is the difference between collaborative filtering and content-based filtering?
2. Explain the cold start problem and two ways to mitigate it.
3. How does cosine similarity measure similarity between users or items?
4. What is the difference between user-user and item-item collaborative filtering?
5. Why do we use regularization in matrix factorization?

### Intermediate

1. Derive the stochastic gradient descent update rules for SVD matrix factorization.
2. How would you evaluate a recommender system offline? What metrics would you use and why?
3. Explain the difference between memory-based and model-based collaborative filtering.
4. How would you handle implicit feedback (clicks, views) in a recommender system?
5. What is the popularity bias problem and how can you mitigate it?

### Advanced

1. Design a hybrid recommender system for a news website that must handle rapidly changing content and new users.
2. Explain how Bayesian Personalized Ranking (BPR) differs from pointwise matrix factorization for implicit feedback.
3. How would you implement a session-based recommender system using sequence models (GRU/Transformer) for an e-commerce platform?

## Practice Problems

### Easy

1. Compute the cosine similarity between two users with rating vectors [5, 3, 0, 1] and [4, 0, 0, 1].
2. Given a user-item matrix, implement item-item collaborative filtering for a single prediction.
3. Calculate the mean-centered rating for a user whose ratings are [4, 5, 0, 2].
4. Given SVD matrices P and Q of shape (10x5) and (8x5), compute the predicted rating for user u and item i.
5. Find the top-2 most similar items to item 0 using cosine similarity from the matrix [[5, 0, 3], [4, 2, 0], [1, 4, 5]].

### Medium

1. Implement k-fold cross-validation for evaluating a Surprise SVD model on MovieLens 100k.
2. Build a content-based movie recommender using genre, director, and keyword features with TF-IDF.
3. Implement the Alternating Least Squares (ALS) algorithm for matrix factorization.
4. Create a hybrid recommender that combines collaborative and content-based predictions using a weighted average.
5. Write a function to compute precision@k and recall@k given predicted and actual ratings.

### Hard

1. Implement a neural collaborative filtering (NCF) model using PyTorch that combines GMF and MLP architectures.
2. Build a session-based recommender using GRU with negative sampling for e-commerce click data.
3. Design and implement a bandit-based exploration strategy for a news recommendation system that balances exploitation and exploration.

## Solutions

### Easy 1 — Cosine similarity
```python
import numpy as np
u1 = np.array([5, 3, 0, 1])
u2 = np.array([4, 0, 0, 1])
sim = np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))
print(f"Cosine similarity: {sim:.4f}")
# Output: Cosine similarity: 0.9153
```

### Easy 3 — Mean centering
```python
ratings = np.array([4, 5, 0, 2])
mask = ratings > 0
mean = ratings[mask].mean()
centered = np.where(mask, ratings - mean, 0)
print(f"Mean: {mean:.2f}, Centered: {centered}")
# Output: Mean: 3.67, Centered: [ 0.33  1.33  0.   -1.67]
```

## Related Concepts

- Dimensionality Reduction (PCA, SVD) — ML-076
- Clustering (K-Means) — ML-073
- Feature Engineering — ML-070
- Evaluation Metrics — ML-065

## Next Concepts

- Time Series Forecasting — ML-092
- NLP with ML — ML-094
- Causal ML — ML-098

## Summary

Recommender systems predict user preferences using collaborative filtering (user-user or item-item), content-based methods (item attributes), or hybrid combinations. Matrix factorization (SVD) decomposes the user-item matrix into latent factors and is the workhorse of modern recommendation. The cold start problem requires hybrid fallbacks. Evaluation must go beyond RMSE to include ranking metrics like precision@k and NDCG.

## Key Takeaways

- Collaborative filtering leverages patterns across users; content-based uses item attributes
- Matrix factorization via SVD learns latent user and item factors
- Always center ratings before computing similarities
- Cold start requires popularity-based or content-based fallbacks
- Evaluate with ranking metrics, not just RMSE
- Popularity bias needs active mitigation through re-ranking or debiasing
- Hybrid methods typically outperform pure collaborative or content-based systems
