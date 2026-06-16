# Concept: Cross Entropy

## Concept ID

MATH-089

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Information Theory

## Learning Objectives

- Define cross entropy as a measure of dissimilarity between two probability distributions
- Understand the relationship $H(P,Q) = H(P) + D_{KL}(P\|Q)$
- Apply cross entropy as a loss function for classification in deep learning
- Distinguish between categorical cross-entropy and binary cross-entropy
- Interpret cross entropy as the average code length when using an incorrect coding distribution

## Prerequisites

- Entropy (MATH-088)
- Probability (MATH-065)
- Random Variable (MATH-070)
- Probability Distribution (MATH-071)
- Logarithms and basic summation

## Definition

**Cross entropy** measures the average number of bits needed to encode outcomes from a true distribution $P$ when using an approximate distribution $Q$ as the coding scheme. For two discrete probability distributions $P$ and $Q$ over the same sample space $\mathcal{X}$, the cross entropy $H(P, Q)$ is defined as:

$$
H(P, Q) = -\sum_{i} P(x_i) \log Q(x_i)
$$

When $P$ is the true distribution and $Q$ is the predicted or estimated distribution, cross entropy quantifies the inefficiency of using $Q$ to represent $P$. It is always at least as large as the entropy of $P$:

$$
H(P, Q) \geq H(P)
$$

with equality if and only if $P = Q$.

In machine learning, cross entropy is **the most important loss function for classification tasks**. Given true labels $y$ (represented as a one-hot distribution $P$) and predicted probabilities $\hat{y}$ (the model output distribution $Q$), the cross-entropy loss measures how well the predictions match the true labels.

## Intuition

Imagine you are designing a code for transmitting messages from a source with known distribution $P$ (e.g., English letter frequencies). If you use the optimal code for $P$, the average code length equals $H(P)$. But if you mistakenly use a code optimised for a different distribution $Q$, the average code length becomes $H(P, Q) \geq H(P)$. The excess over $H(P)$ is the KL divergence $D_{KL}(P\|Q)$.

In the context of classification, the true distribution $P$ is a degenerate distribution placing probability 1 on the correct class (one-hot encoding). The predicted distribution $Q$ is the softmax output. The cross entropy measures how surprised we should be by the true label given the predicted probabilities. If the model assigns high probability to the correct class, cross entropy is low. If it assigns low probability, cross entropy is high — and we want to minimise this.

Cross entropy can also be understood through the lens of likelihood. Minimising cross entropy is equivalent to maximising the log-likelihood of the data under the model. This connection bridges information theory and statistical estimation.

## Why This Concept Matters

Cross entropy is arguably the single most important loss function in modern deep learning:

- **Classification backbone:** Nearly every neural network for classification uses cross-entropy loss, from simple logistic regression to transformer-based language models with millions of classes.
- **Principled optimisation:** Cross entropy is convex in the model parameters for linear models, enabling efficient optimisation. Combined with softmax, it produces well-calibrated probability estimates.
- **Gradient properties:** The gradient of cross-entropy loss with respect to the model's pre-softmax activations (logits) takes a simple form: $\frac{\partial H}{\partial z_k} = \hat{y}_k - y_k$, which is the difference between predicted and true probabilities. This avoids the vanishing gradient problem that plagues squared error with sigmoid outputs.
- **Probabilistic interpretation:** Cross-entropy loss is the negative log-likelihood under a categorical distribution, connecting deep learning to probabilistic modelling.
- **Model comparison:** Lower cross entropy on held-out data indicates a better predictive model, making it a standard evaluation metric.

## Historical Background

The concept of cross entropy emerged naturally from Shannon's information theory. Claude Shannon's 1948 paper established entropy as the fundamental limit of compression. The cross entropy $H(P, Q)$ was understood as the average number of bits required when an incorrect distribution $Q$ is used for coding.

The use of cross entropy as a loss function in machine learning traces to the development of logistic regression in statistics, where maximum likelihood estimation naturally leads to what is now called binary cross-entropy loss. The generalisation to multi-class problems (categorical cross-entropy) followed as neural networks became popular for classification in the 1990s and 2000s.

The key insight connecting cross entropy to neural network training came from the realisation that the softmax function combined with cross-entropy loss produces particularly clean gradients. This combination became standard in deep learning after the success of AlexNet in 2012 and remains the default loss for classification today.

## Real World Examples

1. **Spam detection:** A binary classifier predicts whether an email is spam (1) or not spam (0). The true label for a particular email is "spam" ($y = 1$, so $P = [0, 1]$). The model predicts $\hat{y} = 0.9$ probability of spam ($Q = [0.1, 0.9]$). The binary cross-entropy loss is $-[0 \cdot \log(0.1) + 1 \cdot \log(0.9)] = -\log 0.9 \approx 0.105$. If the model had predicted $\hat{y} = 0.1$, the loss would be $-\log 0.1 \approx 2.303$ — much higher, reflecting the poor prediction.

2. **Image classification:** In ImageNet classification with 1000 classes, the model outputs a vector of 1000 probabilities summing to 1. If the true class is "tabby cat" (index 248) and the model assigns 0.95 probability to that class, the cross-entropy loss for that image is $-\log 0.95 \approx 0.051$. The average cross entropy across the validation set measures overall classification performance.

3. **Language modelling:** A language model predicts the next token in a sequence. For a text corpus, the average cross entropy (perplexity) measures how well the model predicts the text. GPT models are trained to minimise cross-entropy loss over next-token prediction. For a character-level model, cross entropy of 1.5 bits per character means the model is as surprised as if each character came from a distribution with 1.5 bits of uncertainty.

4. **Logistic regression:** In medical diagnosis, a logistic regression model predicts the probability of disease given patient features. The binary cross-entropy loss penalises confident wrong predictions heavily. A model that predicts 95% probability of "no disease" for a patient who actually has the disease incurs a loss of $-\log 0.05 \approx 3.0$.

5. **Multi-label classification:** In a music tagging system where a song can belong to multiple genres simultaneously, binary cross-entropy is applied independently to each genre label. The total loss is the sum of per-genre binary cross-entropies.

## AI/ML Relevance

**Categorical cross-entropy loss:** This is the default loss function for multi-class classification. Given a one-hot encoded true label $y$ (vector of length $K$) and predicted probabilities $\hat{y}$ (softmax output):

$$
\mathcal{L}(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

Since $y$ is one-hot (only one element is 1), this simplifies to $\mathcal{L} = -\log \hat{y}_{\text{true}}$ where $\hat{y}_{\text{true}}$ is the predicted probability of the correct class. Minimising this is equivalent to maximising the log-likelihood of the correct class.

**Binary cross-entropy loss:** Used for binary classification and multi-label classification:

$$
\mathcal{L}(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

This can be seen as categorical cross-entropy with two classes.

**Softmax cross-entropy:** The combination of softmax activation (which converts logits $z$ to probabilities $\hat{y}_k = e^{z_k} / \sum_j e^{z_j}$) followed by cross-entropy loss produces remarkably simple gradients:

$$
\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k
$$

This gradient is simply the prediction error. When the prediction matches the label, the gradient is zero. Otherwise, the gradient pulls logits in the right direction with magnitude proportional to the error.

**Focal loss:** A modification of cross-entropy that downweights well-classified examples, focusing training on hard examples. Focal loss adds a modulating factor $(1-\hat{y}_t)^\gamma$ to the cross-entropy term, where $\gamma \geq 0$.

**Label smoothing:** A regularisation technique that replaces the hard one-hot distribution $y$ with a softened version $y'_k = (1-\epsilon) y_k + \epsilon/K$, where $\epsilon$ is the smoothing parameter. This prevents the model from becoming overconfident and improves generalisation.

**Perplexity:** In language modelling, perplexity is defined as $\text{PPL} = \exp(H(P, Q)) = \exp(-\frac{1}{N}\sum_{t} \log Q(x_t))$, where $x_t$ are the true tokens. Lower perplexity indicates better predictive performance.

## Mathematical Explanation

Cross entropy has deep connections to other information-theoretic quantities. The fundamental relationship is:

$$
H(P, Q) = H(P) + D_{KL}(P\|Q)
$$

Proof:
$$
H(P, Q) = -\sum_i P(i) \log Q(i) = -\sum_i P(i) \log P(i) - \sum_i P(i) \log \frac{Q(i)}{P(i)} = H(P) + \sum_i P(i) \log \frac{P(i)}{Q(i)} = H(P) + D_{KL}(P\|Q)
$$

This decomposition shows that cross entropy equals the intrinsic uncertainty ($H(P)$) plus the additional cost of using the wrong distribution ($D_{KL}(P\|Q)$).

**For continuous distributions:** The cross entropy between two continuous distributions with densities $p$ and $q$ is:

$$
H(p, q) = -\int p(x) \log q(x) \, dx
$$

**For the true distribution being a delta function:** In classification, $P$ is a one-hot distribution ($P(x_j) = 1$ for the true class $j$, 0 otherwise). Then:

$$
H(P, Q) = -\sum_i P(x_i) \log Q(x_i) = -\log Q(x_j)
$$

This is the negative log-likelihood of the true class under the model.

**Relationship to log-likelihood:** For a dataset $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$, the negative log-likelihood under a model $p_\theta(y|x)$ is:

$$
\text{NLL}(\theta) = -\sum_{n=1}^N \log p_\theta(y^{(n)} | x^{(n)})
$$

Normalising by $N$, this is exactly the empirical cross entropy between the empirical distribution of the data and the model distribution. Minimising cross entropy is therefore equivalent to maximum likelihood estimation.

## Formula(s)

**Cross entropy (discrete):**
$$
H(P, Q) = -\sum_{i} P(x_i) \log Q(x_i)
$$

**Categorical cross-entropy loss:**
$$
\mathcal{L}_{CE}(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

**Binary cross-entropy loss:**
$$
\mathcal{L}_{BCE}(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

**Relationship with entropy and KL divergence:**
$$
H(P, Q) = H(P) + D_{KL}(P\|Q)
$$

**Gradient of cross-entropy with softmax:**
$$
\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k
$$

**Perplexity:**
$$
\text{PPL}(P, Q) = 2^{H(P, Q)} \quad (\text{bits}) \quad \text{or} \quad \text{PPL}(P, Q) = \exp(H(P, Q)) \quad (\text{nats})
$$

## Properties

- **Non-negativity:** $H(P, Q) \geq 0$ for any distributions $P, Q$, with equality only when $P$ and $Q$ are both deterministic and agree on the outcome.
- **Lower bound:** $H(P, Q) \geq H(P)$, with equality iff $P = Q$ (by Gibbs' inequality).
- **Not symmetric:** $H(P, Q) \neq H(Q, P)$ in general. Cross entropy is directed: it measures the cost of using $Q$ to approximate $P$.
- **Convexity:** $H(P, Q)$ is convex in $Q$ (for fixed $P$), but not in $P$.
- **Continuous in $Q$:** For fixed $P$, $H(P, Q)$ is continuous in $Q$ (for $Q > 0$).
- **Asymptotic behaviour:** As $Q(x_i) \to 0$ for an outcome where $P(x_i) > 0$, $H(P, Q) \to \infty$. This reflects the infinite cost of using a coding scheme that assigns zero probability to an event that actually occurs.
- **Decomposition:** $H(P, Q) = H(P) + D_{KL}(P\|Q)$.

## Step-by-Step Worked Examples

### Example 1: Binary Cross-Entropy for a Single Prediction

**Problem:** A logistic regression model predicts the probability of a patient having diabetes. For a patient who actually has diabetes ($y = 1$), the model predicts $\hat{y} = 0.7$. Compute the binary cross-entropy loss.

**Solution:**

Step 1: Write the binary cross-entropy formula.
$$
\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

Step 2: Substitute $y = 1$, $\hat{y} = 0.7$.
$$
\mathcal{L} = -[1 \cdot \log 0.7 + 0 \cdot \log 0.3]
$$

Step 3: Simplify. The second term is $0 \cdot \log 0.3 = 0$ (by convention, $0\log 0 = 0$).
$$
\mathcal{L} = -\log 0.7
$$

Step 4: Compute. Using $\ln$ for natural log: $\ln 0.7 \approx -0.3567$. Using $\log_2$: $\log_2 0.7 \approx -0.5146$.
$$
\mathcal{L}_{\text{nats}} = 0.3567
$$
$$
\mathcal{L}_{\text{bits}} = 0.5146
$$

**Answer:** The binary cross-entropy loss is approximately 0.357 nats (or 0.515 bits).

Interpretation: This is a relatively low loss, indicating the model assigned moderately high probability to the correct class.

### Example 2: Categorical Cross-Entropy for Three-Class Classification

**Problem:** A classifier predicts the type of flower among three species: Setosa, Versicolor, Virginica. For a particular flower that is Versicolor (class 2), the model outputs the following softmax probabilities:
$$
\hat{y} = [0.1, 0.8, 0.1]
$$
Compute the categorical cross-entropy loss.

**Solution:**

Step 1: Write the true label as a one-hot vector.
$$
y = [0, 1, 0]
$$

Step 2: Apply categorical cross-entropy.
$$
\mathcal{L} = -\sum_{k=1}^{3} y_k \log \hat{y}_k
$$

Step 3: Substitute.
$$
\mathcal{L} = -[0 \cdot \log 0.1 + 1 \cdot \log 0.8 + 0 \cdot \log 0.1]
$$

Step 4: Simplify.
$$
\mathcal{L} = -\log 0.8 = -\ln 0.8 \approx 0.2231 \text{ nats}
$$

**Answer:** $\mathcal{L} \approx 0.223$ nats (or $-\log_2 0.8 \approx 0.322$ bits).

Note: Only the predicted probability of the true class matters. If the model had predicted $[0.4, 0.3, 0.3]$, the loss would be $-\log 0.3 \approx 1.204$ nats — much higher because the model was less confident in the correct class.

### Example 3: Comparing Two Predicted Distributions

**Problem:** The true distribution of weather for a city is $P = [\text{Sunny: }0.5, \text{Rainy: }0.3, \text{Cloudy: }0.2]$. Two weather forecasters make predictions:
- Forecaster A: $Q_A = [0.4, 0.4, 0.2]$
- Forecaster B: $Q_B = [0.6, 0.2, 0.2]$

Compute the cross entropy for each forecaster and determine who is better.

**Solution:**

Step 1: Compute $H(P, Q_A)$.
$$
H(P, Q_A) = -[0.5\log 0.4 + 0.3\log 0.4 + 0.2\log 0.2]
$$

Using natural logs:
- $\ln 0.4 \approx -0.9163$
- $\ln 0.2 \approx -1.6094$

$$
H(P, Q_A) = -[0.5(-0.9163) + 0.3(-0.9163) + 0.2(-1.6094)]
$$
$$
H(P, Q_A) = -[-0.4582 - 0.2749 - 0.3219] = -[-1.0549] = 1.0549 \text{ nats}
$$

Step 2: Compute $H(P, Q_B)$.
$$
H(P, Q_B) = -[0.5\log 0.6 + 0.3\log 0.2 + 0.2\log 0.2]
$$
- $\ln 0.6 \approx -0.5108$

$$
H(P, Q_B) = -[0.5(-0.5108) + 0.3(-1.6094) + 0.2(-1.6094)]
$$
$$
H(P, Q_B) = -[-0.2554 - 0.4828 - 0.3219] = -[-1.0601] = 1.0601 \text{ nats}
$$

Step 3: Compare. $H(P, Q_A) = 1.0549 < H(P, Q_B) = 1.0601$, so Forecaster A has slightly lower cross entropy and is therefore better.

Step 4: Compute the entropy of $P$ for reference.
$$
H(P) = -[0.5\ln 0.5 + 0.3\ln 0.3 + 0.2\ln 0.2]
$$
- $\ln 0.5 = -0.6931$, $\ln 0.3 = -1.2040$
$$
H(P) = -[0.5(-0.6931) + 0.3(-1.2040) + 0.2(-1.6094)]
$$
$$
H(P) = -[-0.3466 - 0.3612 - 0.3219] = 1.0297 \text{ nats}
$$

The KL divergences are: $D_{KL}(P\|Q_A) = 1.0549 - 1.0297 = 0.0252$, $D_{KL}(P\|Q_B) = 1.0601 - 1.0297 = 0.0304$. Both forecasters are close to the true distribution, but A is slightly more accurate.

**Answer:** Forecaster A is better, with cross entropy 1.0549 nats vs 1.0601 nats for Forecaster B.

### Example 4: Gradient of Cross-Entropy with Softmax

**Problem:** For a 3-class classification problem, the true label is class 2 ($y = [0, 1, 0]$). The model logits are $z = [1.0, 2.0, 0.5]$. Compute the softmax probabilities, cross-entropy loss, and the gradient with respect to each logit.

**Solution:**

Step 1: Compute softmax probabilities.
$$
\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

$$
e^{1.0} = 2.718, \quad e^{2.0} = 7.389, \quad e^{0.5} = 1.649
$$
$$
\sum_j e^{z_j} = 2.718 + 7.389 + 1.649 = 11.756
$$

$$
\hat{y}_1 = 2.718/11.756 = 0.231
$$
$$
\hat{y}_2 = 7.389/11.756 = 0.629
$$
$$
\hat{y}_3 = 1.649/11.756 = 0.140
$$

Step 2: Compute cross-entropy loss.
$$
\mathcal{L} = -\log \hat{y}_2 = -\ln 0.629 = 0.464 \text{ nats}
$$

Step 3: Compute gradients.
$$
\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k
$$
- $\frac{\partial \mathcal{L}}{\partial z_1} = 0.231 - 0 = 0.231$
- $\frac{\partial \mathcal{L}}{\partial z_2} = 0.629 - 1 = -0.371$
- $\frac{\partial \mathcal{L}}{\partial z_3} = 0.140 - 0 = 0.140$

**Answer:** The gradients are $[0.231, -0.371, 0.140]$. The negative gradient for $z_2$ (the true class) indicates that increasing $z_2$ would reduce the loss, while the positive gradients for $z_1$ and $z_3$ indicate that decreasing those logits would also reduce the loss. The magnitudes reflect how much each logit needs to change.

## Visual Interpretation

Cross entropy can be visualised as a function of the predicted probability for the true class. For binary cross-entropy, the loss is $-\log \hat{y}$ when $y=1$, and $-\log(1-\hat{y})$ when $y=0$. This function approaches infinity as the predicted probability approaches 0 (for the correct class), creating a steep penalty for confident wrong predictions. The loss is 0 only when the model is perfectly correct with full confidence.

A contour plot of cross entropy for a 3-class problem (as a function of two softmax inputs) shows a convex landscape with a single global minimum at the true label. This convexity in the logits (not in the parameters of deep networks, but in linear models) makes optimisation well-behaved.

The softmax cross-entropy loss can also be visualised through the lens of logistic regression: it creates a linear decision boundary between each pair of classes, with the loss being proportional to the distance from the boundary on the wrong side.

## Common Mistakes

1. **Confusing cross entropy with KL divergence:** Cross entropy $H(P,Q)$ equals $H(P) + D_{KL}(P\|Q)$. The KL divergence measures the "extra cost" of using $Q$ instead of $P$. Cross entropy includes the base entropy of $P$ as well. In classification, since $H(P) = 0$ for the true one-hot label, cross entropy and KL divergence coincide.

2. **Passing logits directly to cross-entropy without softmax:** Modern deep learning frameworks provide a combined softmax-cross-entropy function (e.g., `tf.nn.softmax_cross_entropy_with_logits` or `torch.nn.CrossEntropyLoss`) that accepts raw logits. Applying softmax manually and then cross entropy can lead to numerical instability when both are implemented separately.

3. **Using the wrong cross-entropy variant:** Binary cross-entropy should be used for binary classification (one output neuron with sigmoid), while categorical cross-entropy should be used for multi-class (multiple output neurons with softmax). Using categorical cross-entropy for binary classification treats it as a 2-class problem with 2 outputs, which works but is less efficient.

4. **Numerical instability with $\log(0)$:** If the model predicts $\hat{y} = 0$ for the true class, $\log 0$ is undefined. Frameworks handle this by adding a small epsilon (e.g., $10^{-7}$) to predictions or by using the log-softmax formulation, which computes $\log(\text{softmax}(z))$ directly without exponentiating intermediate values.

5. **Ignoring class imbalance:** When classes are imbalanced, cross-entropy loss is dominated by the majority class. The model may achieve low loss by always predicting the majority class. Weighted cross-entropy (adding class weights $\alpha_k$: $\mathcal{L} = -\alpha_k \log \hat{y}_k$) addresses this.

6. **Assuming cross entropy is symmetric:** Cross entropy is not symmetric: $H(P,Q) \neq H(Q,P)$ in general. This is because it measures the cost of encoding $P$ using $Q$'s code, which differs from encoding $Q$ using $P$'s code.

7. **Misinterpreting cross-entropy values:** A cross entropy of 0.5 nats does not mean the model is "50% accurate." Cross entropy values depend on the number of classes (baseline cross entropy for random guessing is $\log K$ for $K$ classes). Always compare against baselines.

## Interview Questions

### Beginner - 5

1. **Q:** What is cross entropy?
   **A:** Cross entropy measures the average number of bits needed to encode outcomes from a true distribution $P$ using an approximate distribution $Q$: $H(P,Q) = -\sum P(x)\log Q(x)$.

2. **Q:** What is the relationship between cross entropy, entropy, and KL divergence?
   **A:** $H(P,Q) = H(P) + D_{KL}(P\|Q)$. Cross entropy equals entropy plus KL divergence.

3. **Q:** Why is cross-entropy loss used for classification in deep learning?
   **A:** It measures the dissimilarity between the true label distribution (one-hot) and the predicted distribution (softmax). Minimising cross entropy is equivalent to maximising the log-likelihood, and the softmax-cross-entropy combination produces clean gradients $\hat{y} - y$.

4. **Q:** What is binary cross-entropy loss?
   **A:** It is the cross-entropy loss for binary classification: $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$. It penalises confident wrong predictions heavily.

5. **Q:** When does $H(P,Q) = H(P)$?
   **A:** When $P = Q$ (the predicted distribution matches the true distribution exactly). By Gibbs' inequality, $H(P,Q) \geq H(P)$ with equality iff $P = Q$.

### Intermediate - 5

1. **Q:** How do you compute the gradient of categorical cross-entropy with respect to softmax logits?
   **A:** $\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$, the difference between predicted and true probabilities. This elegant form arises because the softmax normalisation and the cross-entropy logarithm cancel in the derivative.

2. **Q:** What is label smoothing and why is it used?
   **A:** Label smoothing replaces one-hot labels $y_k$ with $y'_k = (1-\epsilon)y_k + \epsilon/K$. This softens the target distribution, preventing the model from becoming overconfident and improving calibration and generalisation.

3. **Q:** How does cross-entropy loss handle multi-label classification?
   **A:** Multi-label classification uses binary cross-entropy independently per label. Each output neuron has a sigmoid activation, and the total loss is the sum of per-label binary cross-entropies. The model can predict multiple classes simultaneously.

4. **Q:** What is the baseline cross-entropy for random guessing in a $K$-class classification problem?
   **A:** If the model predicts uniform probabilities $\hat{y}_k = 1/K$ for all classes, the cross-entropy loss on any true label is $-\log(1/K) = \log K$. For 10-class classification, this is $\log 10 \approx 2.303$ nats. A well-trained model should achieve cross entropy well below this baseline.

5. **Q:** Explain the connection between cross-entropy loss and maximum likelihood estimation.
   **A:** Minimising cross-entropy loss over a dataset is equivalent to maximising the log-likelihood. The negative log-likelihood $\text{NLL} = -\sum_n \log p_\theta(y_n|x_n)$ is exactly the empirical cross entropy. This connection places deep learning on a solid statistical foundation.

### Advanced - 3

1. **Q:** Derive the gradient of binary cross-entropy loss with respect to the logit in logistic regression. Show the full backpropagation chain.
   **A:** Let $z = w^T x + b$ be the logit. The prediction is $\hat{y} = \sigma(z) = 1/(1+e^{-z})$. The loss is $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$. The gradient w.r.t. $z$ is: $\frac{\partial\mathcal{L}}{\partial z} = \frac{\partial\mathcal{L}}{\partial\hat{y}} \cdot \frac{\partial\hat{y}}{\partial z} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y}) = \hat{y} - y$. This is the same elegant form as the multi-class case.

2. **Q:** Analyse the numerical stability issues in computing softmax-cross-entropy. Propose a numerically stable implementation.
   **A:** Computing $\text{softmax}(z_k) = e^{z_k}/\sum_j e^{z_j}$ can overflow for large $z_k$. The numerically stable approach is: (1) subtract $\max_j z_j$ from all logits before exponentiating: $\text{softmax}(z_k) = e^{z_k - m}/\sum_j e^{z_j - m}$ where $m = \max_j z_j$. (2) Compute $\log(\text{softmax}(z_k))$ directly as $z_k - m - \log(\sum_j e^{z_j - m})$ rather than computing softmax first then taking log. Modern frameworks implement this as a fused operation (`log_softmax` followed by `nll_loss`).

3. **Q:** Compare and contrast cross-entropy loss with focal loss for object detection. Under what circumstances is focal loss preferred?
   **A:** Focal loss modifies cross entropy: $\mathcal{L}_{\text{focal}} = -(1-\hat{y}_t)^\gamma \log \hat{y}_t$, where $\hat{y}_t$ is the predicted probability of the true class and $\gamma \geq 0$ (typically 2). When $\gamma=0$, focal loss reduces to cross entropy. The modulating factor $(1-\hat{y}_t)^\gamma$ downweights easy examples (where $\hat{y}_t$ is large) and focuses training on hard, misclassified examples. Focal loss was introduced for dense object detection (RetinaNet) where there is an extreme class imbalance between foreground and background. Cross entropy would be dominated by the vast number of easy negative examples; focal loss prevents this by reducing their contribution.

## Practice Problems

### Easy - 5

1. Compute the binary cross-entropy loss for $y=0$, $\hat{y}=0.1$.

2. In a 5-class classification problem, the true class is class 3, and the model predicts $\hat{y}_3 = 0.6$. What is the categorical cross-entropy loss?

3. True or false: $H(P,Q) \geq H(P)$ always.

4. What is the cross-entropy loss if the predicted probability for the true class is $\hat{y}_{\text{true}} = 1.0$?

5. If $H(P) = 2.0$ nats and $D_{KL}(P\|Q) = 0.5$ nats, what is $H(P,Q)$?

### Medium - 5

1. For a 4-class problem, the true label is $[0, 1, 0, 0]$ and predictions are $[0.2, 0.5, 0.2, 0.1]$. Compute the cross-entropy loss.

2. A binary classifier predicts $\hat{y} = 0.8$ for 3 samples with true labels $[1, 0, 1]$. Compute the average binary cross-entropy loss.

3. Show that $H(P,Q) - H(P,P) = D_{KL}(P\|Q)$.

4. Compute the perplexity (in bits) of a language model with average cross-entropy loss of 0.8 bits per token.

5. A model outputs logits $z = [1.5, 0.5, -0.5]$ for a 3-class problem. The true label is class 1. Compute the softmax probabilities and the cross-entropy loss. Then compute the gradients w.r.t. the logits.

### Hard - 3

1. Prove Gibbs' inequality: $H(P,Q) \geq H(P)$ with equality iff $P = Q$. (Hint: use the log-sum inequality or the fact that $\log x \leq x - 1$.)

2. Derive the class-weighted cross-entropy loss and show its gradient. Explain how it addresses class imbalance.

3. Show that the softmax activation function and cross-entropy loss together produce the same gradient formula regardless of which class is the true class. Generalise this to any distribution $P$ that is not one-hot.

## Solutions

**Easy:**

1. $\mathcal{L} = -[0 \cdot \log 0.1 + 1 \cdot \log 0.9] = -\log 0.9 \approx 0.1054$ nats.

2. $\mathcal{L} = -\log 0.6 \approx 0.5108$ nats.

3. True. $H(P,Q) = H(P) + D_{KL}(P\|Q) \geq H(P)$ because $D_{KL}(P\|Q) \geq 0$.

4. $\mathcal{L} = -\log 1.0 = 0$. The model is perfectly correct with complete confidence.

5. $H(P,Q) = H(P) + D_{KL}(P\|Q) = 2.0 + 0.5 = 2.5$ nats.

**Medium:**

1. $\mathcal{L} = -\log 0.5 = 0.6931$ nats. Only the probability of the true class matters.

2. $\mathcal{L}_1 = -\log 0.8 \approx 0.2231$, $\mathcal{L}_2 = -\log 0.2 \approx 1.6094$, $\mathcal{L}_3 = -\log 0.8 \approx 0.2231$. Average: $(0.2231 + 1.6094 + 0.2231)/3 \approx 0.6852$.

3. $H(P,Q) - H(P,P) = [-\sum P\log Q] - [-\sum P\log P] = -\sum P\log Q + \sum P\log P = \sum P\log(P/Q) = D_{KL}(P\|Q)$.

4. $\text{PPL} = 2^{0.8} \approx 1.741$ — the model is as uncertain as if it had to choose uniformly among 1.74 possible next tokens on average.

5. Softmax: $e^{1.5}=4.4817, e^{0.5}=1.6487, e^{-0.5}=0.6065$, sum $=6.7369$. $\hat{y} = [0.665, 0.245, 0.090]$. $\mathcal{L} = -\log 0.665 \approx 0.408$ nats. Gradients: $[0.665-1, 0.245-0, 0.090-0] = [-0.335, 0.245, 0.090]$.

**Hard:**

1. Gibbs' inequality: Since $\log x \leq x-1$ for $x>0$ (with equality at $x=1$), we have $\log(Q/P) \leq Q/P - 1$. Then $D_{KL}(P\|Q) = -\sum P\log(Q/P) \geq -\sum P(Q/P - 1) = -\sum(Q - P) = -(\sum Q - \sum P) = 0$ since both sum to 1. Therefore $D_{KL}(P\|Q) \geq 0$, so $H(P,Q) = H(P) + D_{KL}(P\|Q) \geq H(P)$. Equality holds when $Q/P = 1$, i.e., $P=Q$.

2. Weighted cross-entropy: $\mathcal{L}_w = -\sum_k \alpha_k y_k \log \hat{y}_k$, where $\alpha_k$ is the weight for class $k$, typically $\alpha_k \propto 1/N_k$ (inverse frequency). The gradient is $\partial\mathcal{L}_w/\partial z_k = \alpha_k(\hat{y}_k - y_k)$ for the true class, and $\partial\mathcal{L}_w/\partial z_j = \alpha_j \hat{y}_j$ for other classes. This increases the penalty for misclassifying rare classes, forcing the model to pay more attention to them.

3. For a one-hot $y$ with $y_c = 1$, $\mathcal{L} = -\log \hat{y}_c$. The gradient $\partial\mathcal{L}/\partial z_k = \hat{y}_k - \delta_{kc}$ (where $\delta_{kc}=1$ if $k=c$, 0 otherwise). This formula holds regardless of $c$. For a non-one-hot $P$, the loss is $\mathcal{L} = -\sum_k P_k \log \hat{y}_k$, and the gradient generalises to $\partial\mathcal{L}/\partial z_k = \hat{y}_k - P_k$. This is intuitive: the gradient pulls $\hat{y}$ toward $P$ componentwise.

## Related Concepts

- Entropy (MATH-088) — the uncertainty of a distribution, and the minimal cross entropy achievable when $Q = P$
- KL Divergence (MATH-090) — $D_{KL}(P\|Q) = H(P,Q) - H(P)$, the excess cost of using $Q$
- Mutual Information (MATH-091) — measure of dependence, related to cross entropy via $I(X;Y) = H(P_{XY}\|P_X P_Y)$
- Information Gain (MATH-092) — reduction in cross entropy after a decision tree split
- Softmax Function — converts logits to probabilities, always paired with cross-entropy loss
- Logistic Regression — uses binary cross-entropy as its natural loss function
- Maximum Likelihood Estimation — cross-entropy minimisation as MLE

## Next Concepts

- KL Divergence (MATH-090) — deepening the relationship between distributions
- Mutual Information (MATH-091) — quantifying dependence
- Information Gain (MATH-092) — applied entropy and cross entropy in decision trees
- Focal Loss — modification for class imbalance
- Perplexity — cross entropy as a model evaluation metric

## Summary

Cross entropy $H(P,Q) = -\sum P(x)\log Q(x)$ measures the average number of bits required to encode samples from distribution $P$ using an optimal code designed for distribution $Q$. It decomposes as $H(P) + D_{KL}(P\|Q)$, showing it includes both the intrinsic uncertainty of $P$ and the excess cost of using the wrong code. In machine learning, cross-entropy loss is the standard loss function for classification, used with softmax activation for multi-class problems and sigmoid for binary problems. The gradient of softmax-cross-entropy is simply $\hat{y} - y$, making optimisation particularly clean. Cross-entropy minimisation is equivalent to maximum likelihood estimation, connecting deep learning to foundational statistical inference.

## Key Takeaways

- $H(P,Q) = -\sum P(x)\log Q(x)$; lower bounded by $H(P)$, equality iff $P=Q$.
- Decomposition: $H(P,Q) = H(P) + D_{KL}(P\|Q)$.
- Classification loss: $\mathcal{L}_{CE} = -\log \hat{y}_{\text{true}}$ for a one-hot true label.
- Gradient with softmax: $\partial\mathcal{L}/\partial z_k = \hat{y}_k - y_k$ — the "error signal."
- Binary cross-entropy: $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$.
- Minimising cross entropy = maximum likelihood estimation.
- Label smoothing, class weighting, and focal loss are important variants.
- Numerical stability requires fused softmax-cross-entropy operations.
