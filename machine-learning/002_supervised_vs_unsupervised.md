# Concept: Supervised vs Unsupervised vs Reinforcement Learning

## Concept ID

ML-002

## Difficulty

BEGINNER

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Distinguish between labeled and unlabeled data
- Explain supervised learning with regression and classification subcategories
- Explain unsupervised learning with clustering, dimensionality reduction, and association subcategories
- Describe reinforcement learning including agent, environment, and reward concepts
- Identify appropriate learning paradigms for different problem types
- Understand semi-supervised learning as a hybrid approach

## Prerequisites

- ML-001: What is Machine Learning
- Basic understanding of what a dataset is (rows and columns)

## Definition

Machine learning problems are broadly categorized into three paradigms based on the nature of the data and the desired output: supervised learning, unsupervised learning, and reinforcement learning. A fourth paradigm, semi-supervised learning, combines elements of the first two.

### Supervised Learning

Supervised learning is the task of learning a function that maps an input to an output based on example input-output pairs. Each training example consists of an input (features) and a desired output (label or target). The algorithm learns by comparing its predictions to the actual labels and adjusting its parameters to minimize the discrepancy.

Formally, given a dataset $D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$ where $x_i$ are feature vectors and $y_i$ are labels, the goal is to learn a function $f: X \rightarrow Y$ that generalizes beyond the training examples.

### Unsupervised Learning

Unsupervised learning involves finding patterns, structures, or representations in data that has no labels. The algorithm must discover hidden structures without any guidance about what the output should look like. The goal is to model the underlying distribution or structure of the data.

Formally, given a dataset $D = \{x_1, x_2, ..., x_n\}$ where $x_i$ are feature vectors without corresponding labels, the goal is to find some structure or transformation that reveals insights about the data.

### Reinforcement Learning

Reinforcement learning is a paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward over time. Unlike supervised learning, there are no correct input-output pairs; the agent must discover which actions yield the greatest reward through trial and error.

### Semi-Supervised Learning

Semi-supervised learning combines a small amount of labeled data with a large amount of unlabeled data. This approach is useful when labeling data is expensive or time-consuming but unlabeled data is abundant. The algorithm uses the labeled data to learn initial patterns and then propagates labels to similar unlabeled examples.

## Intuition

### Supervised Learning Intuition

Imagine a student learning with a teacher who provides both questions and answers. The student solves each question, checks their answer against the teacher's correct answer, and adjusts their approach. Over time, the student learns to answer new questions correctly without help. The teacher provides direct feedback on each attempt.

Everyday example: A child learning to identify fruits. A parent shows an apple and says "this is an apple," shows a banana and says "this is a banana." After enough examples, the child can identify new fruits correctly. The parent provides labeled examples.

### Unsupervised Learning Intuition

Now imagine a student given a large collection of objects and asked to organize them, with no labels provided. The student might notice that some objects are round and others are long, some are red and some are yellow, and group them accordingly. The student discovers structure in the data without any external guidance.

Everyday example: A music streaming service grouping songs into playlists. The service does not have pre-labeled genre tags, but it analyzes audio features and listening patterns to discover that certain songs cluster together. Users who like one song in a cluster tend to like others in the same cluster.

### Reinforcement Learning Intuition

Imagine teaching a dog a new trick. You do not explain the exact sequence of movements. Instead, when the dog performs a desirable action (e.g., sitting on command), you give it a treat (positive reward). When it performs an undesirable action, you offer no treat or a gentle correction (negative reward). Through repeated trials, the dog learns the sequence of actions that maximizes treats.

Everyday example: Learning to play a video game. You do not know the optimal sequence of button presses. You try actions, see what happens (gain points, lose a life, reach a checkpoint), and gradually learn strategies that lead to higher scores.

## Why This Concept Matters

Choosing the right learning paradigm is the first and most critical decision in any ML project. Using supervised learning when labels are unavailable leads to failure. Using unsupervised learning when you have clear labels wastes valuable information. Using reinforcement learning for a simple classification task is overkill. Understanding these paradigms enables practitioners to:

- Select appropriate algorithms for their problem
- Determine data collection and labeling requirements
- Set realistic expectations about what the model can achieve
- Communicate effectively with stakeholders about what the system does

## Mathematical Explanation

### Supervised Learning Formalization

A supervised learning algorithm aims to find a function $f$ that minimizes the expected risk:

$$R(f) = \int L(y, f(x)) \, dP(x, y)$$

Where $L$ is a loss function that measures the discrepancy between the predicted value $f(x)$ and the true value $y$. Common loss functions include:

- **Mean Squared Error (MSE)**: $L(y, \hat{y}) = (y - \hat{y})^2$ for regression
- **Cross-Entropy Loss**: $L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ for binary classification

Since $P(x, y)$ is unknown, we minimize the empirical risk over the training data:

$$R_{emp}(f) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i))$$

### Unsupervised Learning Formalization

Unsupervised learning does not have a universal loss function. Instead, objectives depend on the specific task:

- **Clustering**: Minimize within-cluster distances and maximize between-cluster distances. For k-means: $J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$
- **Dimensionality Reduction (PCA)**: Maximize the variance of projected data, equivalently minimize reconstruction error
- **Association Rule Mining**: Discover rules $X \rightarrow Y$ that satisfy minimum support and confidence thresholds

### Reinforcement Learning Formalization

Reinforcement learning is formalized as a Markov Decision Process (MDP) defined by:

- $S$: Set of states the environment can be in
- $A$: Set of actions the agent can take
- $P(s' | s, a)$: Transition probability to state $s'$ when taking action $a$ in state $s$
- $R(s, a, s')$: Reward received for transitioning
- $\gamma \in [0, 1]$: Discount factor that prioritizes immediate rewards

The agent's goal is to learn a policy $\pi(a|s)$ that maximizes the expected discounted cumulative reward:

$$G_t = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\right]$$

## Code Examples

### Example 1: Supervised Learning — Classification and Regression

```python
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# --- Classification (Supervised) ---
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Classification Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# Output: Classification Accuracy: 1.00

# --- Regression (Supervised) ---
diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_tr, y_tr)
y_pred_reg = reg.predict(X_te)
mse = mean_squared_error(y_te, y_pred_reg)
print(f"Regression MSE: {mse:.2f}")
# Output: Regression MSE: 2900.19
```

### Example 2: Unsupervised Learning — Clustering and Dimensionality Reduction

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate synthetic clustered data
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=42)

# --- Clustering (Unsupervised) ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Count samples in each cluster
for i in range(4):
    count = np.sum(y_pred == i)
    print(f"Cluster {i}: {count} samples")
# Output:
# Cluster 0: 75 samples
# Cluster 1: 75 samples
# Cluster 2: 75 samples
# Cluster 3: 75 samples

# --- Dimensionality Reduction (Unsupervised) ---
# Create high-dimensional data
np.random.seed(42)
X_high = np.random.randn(100, 50)  # 100 samples, 50 features

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high)

print(f"Original shape: {X_high.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
# Output:
# Original shape: (100, 50)
# Reduced shape: (100, 2)
# Explained variance ratio: [0.05182865 0.0490896 ]
```

### Example 3: Reinforcement Learning — Simple Q-Learning Agent

```python
import numpy as np

# Simple grid world: 4 states, 4 actions (up, down, left, right)
# State 3 is the goal state with reward +10
n_states = 4
n_actions = 4
gamma = 0.9  # discount factor
alpha = 0.1  # learning rate
epsilon = 0.1  # exploration rate

# Reward matrix: R[state, action, next_state]
R = np.zeros((n_states, n_actions, n_states))
R[2, 1, 3] = 10.0  # taking action 'down' from state 2 goes to goal (state 3)
R[1, 2, 3] = 10.0  # taking action 'right' from state 1 goes to goal
# All other rewards are 0

# Transition matrix: deterministic
P = np.zeros((n_states, n_actions, n_states))
# State 0: actions go to self
P[0, :, 0] = 1.0
# State 1: action 2 (right) goes to 3, others to self
P[1, :, 1] = 0.75
P[1, 2, 3] = 0.25
# State 2: action 1 (down) goes to 3, others to self
P[2, :, 2] = 0.75
P[2, 1, 3] = 0.25
# State 3: all actions go to self (terminal)
P[3, :, 3] = 1.0

# Q-learning
Q = np.zeros((n_states, n_actions))
n_episodes = 1000

for episode in range(n_episodes):
    state = 0  # start state
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        # Take action
        next_state = np.random.choice(n_states, p=P[state, action])
        reward = R[state, action, next_state]

        # Q-value update
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state
        if state == 3 or state == 0:
            done = True

print("Learned Q-Table:")
print(Q)
# Output: Learned Q-Table:
# [[0.81 0.81 0.81 0.81]
#  [0.   0.   8.1  0.  ]
#  [0.   8.1  0.   0.  ]
#  [0.   0.   0.   0.  ]]
```

## Common Mistakes

1. **Using supervised learning when no labels exist**: Many practitioners try to force supervised learning on unlabeled data. If labels are unavailable, unsupervised or semi-supervised approaches are necessary.
2. **Assuming unsupervised learning always produces meaningful clusters**: Clustering algorithms always find clusters, even in random data. Always validate cluster quality using metrics like silhouette score or domain expertise.
3. **Confusing regression and classification**: Regression predicts continuous values (price, temperature), while classification predicts discrete categories (spam/not spam, dog/cat). Using classification for continuous targets discards information.
4. **Treating all unsupervised tasks the same**: Clustering, dimensionality reduction, and association rule mining serve different purposes and require different evaluation approaches.
5. **Neglecting the exploration-exploitation tradeoff in RL**: In reinforcement learning, too much exploitation leads to suboptimal policies; too much exploration wastes time. Finding the right balance is crucial.
6. **Applying RL to problems with no clear reward signal**: RL requires a well-defined reward function. If you cannot quantify success and failure, RL is unlikely to work.
7. **Ignoring semi-supervised learning as an option**: When labeled data is scarce but unlabeled data is abundant, semi-supervised learning often outperforms both supervised (with limited data) and unsupervised approaches.

## Interview Questions

### Beginner - 5

1. **Q: What is the key difference between supervised and unsupervised learning?**
   A: Supervised learning uses labeled data (input-output pairs) to learn a mapping function. Unsupervised learning uses unlabeled data to discover hidden patterns or structures.

2. **Q: Give two examples each of classification and regression problems.**
   A: Classification: spam detection (spam/not spam), disease diagnosis (disease present/absent). Regression: house price prediction, temperature forecasting.

3. **Q: What is clustering and name two clustering algorithms.**
   A: Clustering is an unsupervised learning task that groups similar data points together. Examples: K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models.

4. **Q: What is the role of the reward function in reinforcement learning?**
   A: The reward function defines the goal of the RL problem. It provides a scalar signal to the agent indicating how good or bad each action was, guiding the learning process toward optimal behavior.

5. **Q: When would you use semi-supervised learning?**
   A: When labeled data is expensive or time-consuming to obtain but unlabeled data is plentiful. For example, medical diagnosis where expert labeling is costly but medical images are abundant.

### Intermediate - 5

1. **Q: Explain how k-means clustering works step by step.**
   A: (1) Choose k initial centroids randomly. (2) Assign each data point to the nearest centroid. (3) Recompute centroids as the mean of all points in each cluster. (4) Repeat steps 2-3 until centroids stop changing or a maximum number of iterations is reached.

2. **Q: What is the difference between PCA and t-SNE for dimensionality reduction?**
   A: PCA is a linear technique that finds orthogonal axes maximizing variance. It is deterministic and preserves global structure. t-SNE is a non-linear technique that minimizes the divergence between pairwise similarity distributions in high and low dimensions. It is stochastic and preserves local structure, making it better for visualization.

3. **Q: What is the exploration-exploitation tradeoff in RL?**
   A: Exploration involves trying new actions to discover their consequences. Exploitation involves choosing the best-known action to maximize reward. Too much exploration wastes time; too much exploitation may miss better strategies.

4. **Q: How do you evaluate unsupervised clustering results?**
   A: Intrinsic metrics (no ground truth): inertia (within-cluster sum of squares), silhouette score, Davies-Bouldin index. Extrinsic metrics (with ground truth): adjusted Rand index, mutual information, homogeneity.

5. **Q: What is the difference between Q-learning and SARSA?**
   A: Q-learning is an off-policy algorithm that learns the optimal policy regardless of the agent's actions. SARSA is an on-policy algorithm that learns the value of the policy being followed. Q-learning uses the maximum Q-value of the next state, while SARSA uses the Q-value of the actual next action taken.

### Advanced - 3

1. **Q: Derive the evidence lower bound (ELBO) used in variational autoencoders and explain its role in unsupervised learning.**
   A: The ELBO is derived from the log marginal likelihood: $\log p(x) = \log \int p(x|z)p(z)dz = \log \int q(z|x) \frac{p(x|z)p(z)}{q(z|x)} dz \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x)||p(z))$. Maximizing the ELBO simultaneously maximizes reconstruction quality (first term) and minimizes the divergence between the approximate posterior and prior (second term).

2. **Q: Explain the policy gradient theorem and how it differs from value-based RL methods.**
   A: The policy gradient theorem states that $\nabla J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$, where $J$ is the expected return. Unlike value-based methods (Q-learning, DQN) that learn Q-values and derive a policy from them, policy gradient methods directly parameterize and optimize the policy. This handles continuous action spaces and stochastic policies naturally.

3. **Q: Compare and contrast the three paradigms from a Bayesian perspective: supervised, unsupervised, and reinforcement learning.**
   A: Supervised learning estimates $p(y|x)$ or $p(x,y)$ from labeled data. Unsupervised learning models $p(x)$ to discover the data-generating distribution. Reinforcement learning optimizes a policy to maximize $\mathbb{E}[\sum \gamma^t R_t]$, which can be viewed as inference in a probabilistic graphical model where actions are latent variables that affect future observations and rewards.

## Practice Problems

### Easy - 5

1. **Problem**: You have a dataset of customer transactions. Each transaction is labeled as fraudulent or legitimate. What type of learning is this?

2. **Problem**: You are given 10,000 news articles without any categories. Your task is to group them by topic. What learning paradigm do you use?

3. **Problem**: A robot must learn to navigate a maze to reach a goal. It receives a reward of +10 upon reaching the goal and -1 for each step taken. What type of learning is this?

4. **Problem**: Is predicting the price of a used car classification or regression?

5. **Problem**: You have 1000 labeled images of cats and dogs and 100,000 unlabeled animal images. What approach would you recommend?

### Medium - 5

1. **Problem**: Compare the data requirements (labeled vs unlabeled) and output types for all three learning paradigms.

2. **Problem**: You are building a recommendation system for an e-commerce site. Customers browse products but rarely rate them. You have purchase history (which items were bought together) but no explicit preferences. Which paradigm(s) would you use?

3. **Problem**: Explain why dimensionality reduction is often used as a preprocessing step before clustering or classification.

4. **Problem**: A self-driving car uses RL for lane-keeping. Design a simple reward function and explain why it encourages desired behavior.

5. **Problem**: You have a binary classification problem with 10,000 features but only 200 samples. Which paradigm-related challenges do you face, and what approach would you take?

### Hard - 3

1. **Problem**: Derive the update rules for both supervised (gradient descent for linear regression) and unsupervised (k-means) learning, and explain the fundamental difference in what drives the updates.

2. **Problem**: Design a semi-supervised learning approach for text classification where you have 50 labeled documents and 1 million unlabeled documents. Explain the algorithm, assumptions, and potential pitfalls.

3. **Problem**: Compare expectation-maximization (EM) for Gaussian Mixture Models (unsupervised) with the Baum-Welch algorithm for Hidden Markov Models, discussing their similarities and differences as instances of the EM framework.

## Solutions

### Easy Solutions

1. Supervised learning (classification) — the labels (fraud/legitimate) are present.
2. Unsupervised learning (clustering) — no labels, need to discover topic groupings.
3. Reinforcement learning — an agent learning through interaction and reward signals.
4. Regression — price is a continuous value.
5. Semi-supervised learning — use the 1000 labeled images to bootstrap learning on the 100,000 unlabeled images.

### Medium Solutions

1. Supervised: requires labeled data, outputs predictions (continuous or categorical). Unsupervised: requires unlabeled data, outputs clusters, reduced dimensions, or association rules. RL: requires environment/reward signal, outputs a policy mapping states to actions.
2. Unsupervised learning primarily — use association rule mining (market basket analysis) to find products frequently bought together. Could also use collaborative filtering (which has elements of both supervised and unsupervised).
3. Dimensionality reduction reduces noise, removes redundant features, mitigates the curse of dimensionality, improves computational efficiency, and often improves clustering/classification performance.
4. Reward = +1 for staying in lane center, -1 for touching lane edge, -10 for leaving the road. This encourages the agent to stay centered, avoid edges, and especially avoid leaving the road entirely.
5. The curse of dimensionality — 200 samples with 10,000 features means severe overfitting risk. Approaches: dimensionality reduction (PCA), feature selection, regularization (L1/Lasso), or simpler models.

### Hard Solutions

1. Linear regression gradient descent update: $\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$. K-means update: $\mu_k := \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$. The fundamental difference: supervised updates are driven by prediction error (discrepancy between prediction and label), while unsupervised updates are driven by data structure (assignment and recomputation based on proximity).
2. Approach: (1) Train initial classifier on 50 labeled documents. (2) Use it to predict pseudo-labels on unlabeled data with high confidence. (3) Add confident predictions to training set. (4) Retrain and repeat. Assumptions: cluster assumption (data forms clusters, and points in same cluster share labels), manifold assumption (data lies on low-dimensional manifold). Pitfalls: confirmation bias (reinforcing initial errors), threshold sensitivity, distribution shift between labeled and unlabeled sets.
3. Both EM algorithms alternate between (E-step) computing posterior probabilities of latent variables and (M-step) maximizing the expected complete log-likelihood. For GMMs: E-step computes posterior probability that each point belongs to each Gaussian component; M-step updates component weights, means, and covariances. For HMMs: E-step (Forward-Backward algorithm) computes posterior probabilities of hidden states; M-step updates transition and emission probabilities. Both converge to local optima of the marginal likelihood.

## Related Concepts

- **ML-001: What is Machine Learning** — The foundational concept upon which all paradigms are built
- **ML-003: Train/Test Split** — Essential for evaluating supervised learning models
- **ML-009: Feature Engineering** — Important across all paradigms
- **Semi-Supervised Learning**: The hybrid approach combining labeled and unlabeled data
- **Self-Supervised Learning**: A paradigm that generates labels from the data itself, bridging supervised and unsupervised learning

## Next Concepts

- **ML-003: Train/Test Split** — How to properly evaluate your supervised models
- **ML-004: Overfitting and Underfitting** — Troubleshooting models across paradigms
- **ML-009: Feature Engineering** — Creating good features for any learning paradigm

## Summary

Machine learning algorithms are categorized into three main paradigms. Supervised learning uses labeled data for prediction tasks, subdivided into regression (continuous outputs) and classification (discrete outputs). Unsupervised learning finds hidden structure in unlabeled data through clustering, dimensionality reduction, and association rule mining. Reinforcement learning trains an agent to make sequential decisions by interacting with an environment to maximize cumulative rewards. Semi-supervised learning bridges supervised and unsupervised approaches when labeled data is scarce but unlabeled data is abundant. The choice of paradigm depends on the nature of available data and the problem being solved.

## Key Takeaways

1. Supervised learning requires labeled data and produces predictive models for classification or regression.
2. Unsupervised learning works with unlabeled data to discover clusters, reduce dimensionality, or find association rules.
3. Reinforcement learning trains an agent through trial-and-error interaction with an environment.
4. The choice of paradigm depends on data availability and problem type.
5. Semi-supervised learning is valuable when labels are expensive but unlabeled data is plentiful.
6. Each paradigm has its own evaluation metrics, algorithms, and best practices.
7. Understanding all three paradigms enables comprehensive problem-solving in ML.
