# Concept: Bernoulli Distribution

## Concept ID

MATH-072

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

1. Define the Bernoulli distribution and its probability mass function
2. Compute the mean, variance, and moment-generating function of a Bernoulli random variable
3. Distinguish between a Bernoulli trial and a Bernoulli distribution
4. Apply the Bernoulli distribution to model binary outcomes in real-world scenarios
5. Connect the Bernoulli distribution to logistic regression and binary cross-entropy loss in machine learning

## Prerequisites

- Basic set theory and sample spaces
- Definition of a random variable (discrete vs continuous)
- Concept of probability mass function (PMF)
- Expected value and variance of a discrete random variable
- Elementary algebra and combinatorics

## Definition

The Bernoulli distribution is the discrete probability distribution of a random variable that takes the value $1$ with probability $p$ and the value $0$ with probability $q = 1-p$, where $0 \leq p \leq 1$. It is named after the Swiss mathematician Jacob Bernoulli. A random variable $X$ following a Bernoulli distribution is denoted as

$$X \sim \text{Bernoulli}(p)$$

The probability mass function (PMF) is given by

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

Equivalently,

$$P(X = 1) = p, \quad P(X = 0) = 1-p$$

## Intuition

The Bernoulli distribution models any experiment that has exactly two outcomes: success and failure. The classic example is a single coin flip. If we define "heads" as success ($X=1$) and "tails" as failure ($X=0$), and the coin is fair, then $p = 0.5$. The distribution is the simplest non-trivial probability distribution because it has only two possible outcomes, yet it serves as the building block for many other distributions, most notably the Binomial distribution.

Think of a Bernoulli random variable as an indicator that a specific event occurred. For instance, "Did this email arrive?" or "Did this customer click the ad?" or "Does this patient have the disease?" Each such yes/no question, when the outcome is uncertain, can be modelled with a Bernoulli random variable.

## Why This Concept Matters

The Bernoulli distribution is the fundamental building block of discrete probability theory. Understanding it is essential because:

- It is the simplest distribution that captures randomness in a binary outcome
- It forms the basis for the Binomial, Geometric, and Negative Binomial distributions
- It directly models classification problems in machine learning where the output is binary
- Logistic regression, one of the most widely used classification algorithms, models the probability $p$ of a Bernoulli outcome as a function of input features
- Binary cross-entropy loss, the standard loss function for binary classification, derives directly from the Bernoulli likelihood

## Historical Background

The distribution is named after Jacob Bernoulli (1655--1705), one of the prominent members of the Bernoulli family of mathematicians. Jacob Bernoulli made fundamental contributions to probability theory and calculus of variations. His seminal work *Ars Conjectandi* (The Art of Conjecturing), published posthumously in 1713, contained the Bernoulli distribution, the Bernoulli trial concept, and the first version of the Law of Large Numbers (the Bernoulli theorem). The concept of a Bernoulli trial — an experiment with exactly two outcomes — became a cornerstone of probability theory and is still taught in every introductory probability course worldwide.

## Real World Examples

1. **Medical Testing**: A rapid antigen test for a disease produces either a positive or negative result. If the test has 95% sensitivity, then for an infected patient, the result $X$ (1 = positive, 0 = negative) follows $X \sim \text{Bernoulli}(0.95)$.

2. **Email Classification**: An email is classified as spam (1) or not spam (0). For a particular filter, the probability that a given spam email is correctly flagged is $p = 0.98$.

3. **Manufacturing Quality Control**: A factory produces light bulbs. Each bulb is tested and classified as defective (1) or non-defective (0). If the historical defect rate is 2%, then each bulb's status is $X \sim \text{Bernoulli}(0.02)$.

4. **Customer Conversion**: An e-commerce website shows an ad to a visitor. The visitor either clicks (1) or does not click (0). If the click-through rate is 3%, then $X \sim \text{Bernoulli}(0.03)$.

5. **Sports**: In basketball, a free throw is either made (1) or missed (0). For a player with a 75% free-throw percentage, each free throw is $X \sim \text{Bernoulli}(0.75)$.

## AI/ML Relevance

The Bernoulli distribution is central to binary classification, which is one of the most common types of supervised learning problems.

**Binary Classification Output**: In binary classification, the model outputs a probability $\hat{p} \in [0, 1]$ that the input belongs to the positive class. The true label $y \in \{0, 1\}$ is assumed to be a Bernoulli random variable with parameter $p$ (the true underlying probability). The model's job is to estimate $p$ as accurately as possible.

**Logistic Regression**: Logistic regression models the log-odds of the probability $p$ as a linear function of the features:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n$$

Solving for $p$ gives the logistic (sigmoid) function:

$$p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}$$

This $p$ is the parameter of the Bernoulli distribution governing the label $y$.

**Binary Cross-Entropy Loss**: Given a dataset of $N$ examples $\{(x_i, y_i)\}$ where $y_i \in \{0, 1\}$ and the model predicts $\hat{p}_i = P(y_i = 1 \mid x_i)$, the likelihood of the data under the Bernoulli model is

$$\mathcal{L} = \prod_{i=1}^{N} \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

Taking the negative logarithm yields the binary cross-entropy loss:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

This is the standard loss function for binary classification in neural networks, including models like BERT for text classification, ResNet for image classification, and virtually all binary classifiers in deep learning.

**Naive Bayes Classifier**: The Bernoulli Naive Bayes classifier explicitly assumes that each feature is a Bernoulli random variable, making it suitable for binary feature vectors like those arising from text classification with bag-of-words representations.

**A/B Testing**: In A/B testing, the conversion rate of a control group and a treatment group are each modelled as Bernoulli random variables. Statistical tests compare whether the difference in conversion rates is significant.

## Mathematical Explanation

### Probability Mass Function

The PMF of $X \sim \text{Bernoulli}(p)$ is:

$$P(X = x) = \begin{cases} p & \text{if } x = 1 \\ 1 - p & \text{if } x = 0 \end{cases}$$

A compact form is:

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

### Cumulative Distribution Function

The CDF is:

$$F(x) = P(X \leq x) = \begin{cases} 0 & x < 0 \\ 1 - p & 0 \leq x < 1 \\ 1 & x \geq 1 \end{cases}$$

### Expected Value

$$\mathbb{E}[X] = \sum_{x \in \{0, 1\}} x \cdot P(X = x) = 0 \cdot (1-p) + 1 \cdot p = p$$

### Variance

$$\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

Since $X^2 = X$ for $X \in \{0, 1\}$ (because $0^2 = 0$ and $1^2 = 1$), we have $\mathbb{E}[X^2] = \mathbb{E}[X] = p$. Therefore:

$$\text{Var}[X] = p - p^2 = p(1-p)$$

### Moment-Generating Function

$$M_X(t) = \mathbb{E}[e^{tX}] = e^{t \cdot 0} \cdot (1-p) + e^{t \cdot 1} \cdot p = 1 - p + p e^{t}$$

### Skewness and Kurtosis

The skewness of $\text{Bernoulli}(p)$ is:

$$\gamma_1 = \frac{1 - 2p}{\sqrt{p(1-p)}}$$

The excess kurtosis is:

$$\gamma_2 = \frac{1 - 6p(1-p)}{p(1-p)}$$

When $p = 0.5$, the distribution is symmetric (skewness = 0). When $p < 0.5$, skewness is positive; when $p > 0.5$, skewness is negative.

## Formula(s)

1. **PMF**: $P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$

2. **Mean**: $\mathbb{E}[X] = p$

3. **Variance**: $\text{Var}[X] = p(1-p)$

4. **MGF**: $M_X(t) = 1 - p + p e^{t}$

5. **CDF**: $F(x) = \begin{cases} 0 & x < 0 \\ 1 - p & 0 \leq x < 1 \\ 1 & x \geq 1 \end{cases}$

## Properties

1. **Sum of Bernoulli variables**: If $X_1, X_2, \ldots, X_n$ are independent and identically distributed $\text{Bernoulli}(p)$ random variables, then $\sum_{i=1}^n X_i \sim \text{Binomial}(n, p)$.

2. **Indicator variables**: A Bernoulli random variable can represent the indicator of any event $A$, i.e., $X = \mathbb{I}_A$ where $X = 1$ if $A$ occurs and $X = 0$ otherwise. In this case, $\mathbb{E}[\mathbb{I}_A] = P(A)$.

3. **Maximum entropy**: Among all discrete distributions supported on $\{0, 1\}$ with a given mean $p$, the Bernoulli distribution with parameter $p$ is the maximum entropy distribution.

4. **Exponential family**: The Bernoulli distribution belongs to the exponential family of distributions. Its natural parameter is $\theta = \log(p/(1-p))$, which is the log-odds.

5. **Reproductivity**: The Bernoulli distribution is not reproductive under addition (the sum of Bernoulli variables is Binomial, not Bernoulli), but it is reproductive under the operation of taking independent copies conditioned on the sum (this relates to exchangeability).

6. **Conjugate prior**: The Beta distribution is the conjugate prior for the parameter $p$ of a Bernoulli distribution. If $p \sim \text{Beta}(\alpha, \beta)$ and $X \mid p \sim \text{Bernoulli}(p)$, then the posterior is $p \mid X \sim \text{Beta}(\alpha + X, \beta + 1 - X)$.

## Step-by-Step Worked Examples

### Example 1: Fair Coin Flip

A fair coin is flipped once. Let $X = 1$ if heads appears, $X = 0$ otherwise.

**Step 1**: Identify the parameter. Since the coin is fair, $p = 0.5$.

**Step 2**: Write the distribution. $X \sim \text{Bernoulli}(0.5)$.

**Step 3**: Compute the probability of heads.

$$P(X = 1) = p = 0.5$$

**Step 4**: Compute the probability of tails.

$$P(X = 0) = 1 - p = 0.5$$

**Step 5**: Compute the expected value.

$$\mathbb{E}[X] = p = 0.5$$

**Step 6**: Compute the variance.

$$\text{Var}[X] = p(1-p) = 0.5 \times 0.5 = 0.25$$

**Interpretation**: Over many flips, the average outcome is 0.5, meaning half heads and half tails. The variance of 0.25 quantifies the spread of outcomes around this mean.

### Example 2: Disease Testing

A disease test has 99% sensitivity (correctly identifies diseased patients). A randomly selected diseased patient is tested. Let $X = 1$ if the test is positive, $X = 0$ otherwise.

**Step 1**: Identify the parameter. Sensitivity means $P(\text{positive} \mid \text{diseased}) = 0.99$, so $p = 0.99$.

**Step 2**: Write the distribution. $X \sim \text{Bernoulli}(0.99)$.

**Step 3**: Compute the probability of a positive test.

$$P(X = 1) = 0.99$$

**Step 4**: Compute the probability of a false negative.

$$P(X = 0) = 1 - 0.99 = 0.01$$

**Step 5**: Compute the mean and variance.

$$\mathbb{E}[X] = 0.99, \quad \text{Var}[X] = 0.99 \times 0.01 = 0.0099$$

**Step 6**: Interpret. The test correctly detects the disease 99% of the time. The variance is very small because $p$ is close to 1, meaning the outcome is highly predictable.

### Example 3: Click-Through Rate (CTR) and Binary Cross-Entropy

An ad campaign has a historical click-through rate of 2%. A logistic regression model predicts $p = 0.03$ for a particular user. The user does not click ($y = 0$).

**Step 1**: The true outcome is $y \sim \text{Bernoulli}(0.02)$ (assuming the historical rate is the true probability). The model predicts $\hat{p} = 0.03$.

**Step 2**: Compute the likelihood of this observation under the model's prediction.

$$P(y = 0 \mid \hat{p} = 0.03) = 1 - 0.03 = 0.97$$

**Step 3**: Compute the contribution to the binary cross-entropy loss.

$$\text{Loss} = -\left[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]$$
$$\text{Loss} = -\left[ 0 \cdot \log(0.03) + 1 \cdot \log(0.97) \right]$$
$$\text{Loss} = -\log(0.97) \approx 0.0305$$

**Step 4**: Interpret. The loss is small (close to 0), which makes sense because the model predicted a low probability of a click, and indeed no click occurred.

**Step 5**: Now suppose the same model predicts $\hat{p} = 0.9$ for another user (anomalous prediction), and again the user does not click.

$$\text{Loss} = -\log(0.1) \approx 2.3026$$

**Step 6**: Interpret. The loss is large, indicating a poor prediction. The model was very confident the user would click (90% confidence) but was wrong. This showcases why cross-entropy heavily penalizes confident but incorrect predictions.

### Example 4: Parameter Estimation via Maximum Likelihood

We observe 10 independent Bernoulli trials: 1, 0, 1, 1, 0, 1, 1, 0, 1, 1. Estimate $p$ using maximum likelihood estimation (MLE).

**Step 1**: Count successes. There are 7 ones and 3 zeros.

**Step 2**: Write the likelihood function.

$$L(p) = \prod_{i=1}^{10} p^{x_i} (1-p)^{1-x_i} = p^7 (1-p)^3$$

**Step 3**: Take the log-likelihood.

$$\ell(p) = 7 \log p + 3 \log(1-p)$$

**Step 4**: Differentiate and set to zero.

$$\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0$$
$$\frac{7}{p} = \frac{3}{1-p}$$
$$7(1-p) = 3p$$
$$7 - 7p = 3p$$
$$7 = 10p$$
$$p = 0.7$$

**Step 5**: The MLE is $\hat{p} = 0.7$, which equals the sample proportion of successes.

**Step 6**: Compute the standard error.

$$\text{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.7 \times 0.3}{10}} \approx 0.1449$$

### Example 5: Bayesian Updating with Beta Prior

A new drug is being tested. Based on prior knowledge, the success probability $p$ follows $\text{Beta}(2, 8)$ (mean = 0.2). In a trial of 1 patient, the drug succeeds. Find the posterior distribution.

**Step 1**: Prior: $p \sim \text{Beta}(\alpha = 2, \beta = 8)$.

Prior mean: $\frac{\alpha}{\alpha + \beta} = \frac{2}{10} = 0.2$.

**Step 2**: Likelihood of the observation $X = 1$ given $p$: $P(X = 1 \mid p) = p$.

**Step 3**: Posterior is proportional to prior times likelihood.

$$f(p \mid X = 1) \propto f(p) \cdot P(X = 1 \mid p)$$
$$\propto p^{\alpha - 1} (1-p)^{\beta - 1} \cdot p$$
$$\propto p^{\alpha} (1-p)^{\beta - 1}$$

**Step 4**: Identify this as $\text{Beta}(\alpha + 1, \beta)$.

Posterior: $p \mid X = 1 \sim \text{Beta}(3, 8)$.

**Step 5**: Posterior mean: $\frac{3}{11} \approx 0.2727$.

**Interpretation**: After seeing one success, the estimated success probability increased from 0.2 to 0.2727.

## Visual Interpretation

The Bernoulli distribution can be visualized as a simple bar chart with two bars:

- A bar at $x = 0$ with height $1-p$
- A bar at $x = 1$ with height $p$

When $p = 0.5$, both bars are equal height, representing a symmetric distribution. When $p > 0.5$, the bar at $x = 1$ is taller, and the distribution is skewed left (negative skew). When $p < 0.5$, the bar at $x = 0$ is taller, and the distribution is skewed right (positive skew).

The expected value $p$ is the balance point of this bar chart. It can be thought of as the "centre of mass" of the two-point distribution.

## Common Mistakes

1. **Confusing Bernoulli with Binomial**: The Bernoulli distribution models one trial; the Binomial models the sum of $n$ independent Bernoulli trials. A single Bernoulli($p$) variable is equivalent to Binomial($1, p$).

2. **Assuming $p = 0.5$ by default**: Many beginners assume $p = 0.5$ when no probability is given. The parameter $p$ must always be specified or estimated from data.

3. **Misinterpreting the variance formula**: The variance $p(1-p)$ is maximized when $p = 0.5$, giving $0.25$. Students sometimes think variance increases as $p$ approaches 1, but it actually approaches 0 because the outcome becomes nearly certain.

4. **Confusing $p$ with the probability of $X = 0$**: $P(X = 0) = 1-p$, not $p$. The parameter $p$ is always the probability of success ($X = 1$), not failure.

5. **Applying Bernoulli to non-binary outcomes**: The Bernoulli distribution only applies to experiments with exactly two outcomes. Problems with three or more categories (e.g., red/green/blue) require the Categorical distribution.

6. **Assuming independence when it does not hold**: Bernoulli trials must be independent. Drawing cards without replacement from a deck produces dependent Bernoulli variables, which follow a Hypergeometric distribution, not Bernoulli.

7. **Treating the loss function as the distribution**: Binary cross-entropy is derived from the Bernoulli likelihood but is not itself a probability distribution. The two concepts are related but distinct.

## Interview Questions

### Beginner

1. **Q**: What is the support of a Bernoulli random variable?
   **A**: The support is $\{0, 1\}$. The variable can only take the value 0 or 1.

2. **Q**: If $X \sim \text{Bernoulli}(0.3)$, what is $P(X = 0)$?
   **A**: $P(X = 0) = 1 - p = 1 - 0.3 = 0.7$.

3. **Q**: What is the expected value of a Bernoulli($p$) random variable?
   **A**: $\mathbb{E}[X] = p$.

4. **Q**: What is the variance of a Bernoulli($p$) random variable?
   **A**: $\text{Var}[X] = p(1-p)$.

5. **Q**: When is the variance of a Bernoulli distribution maximized?
   **A**: At $p = 0.5$, where $\text{Var}[X] = 0.25$.

### Intermediate

1. **Q**: Derive the moment-generating function of a Bernoulli($p$) random variable.
   **A**: $M_X(t) = \mathbb{E}[e^{tX}] = e^{0} \cdot (1-p) + e^{t} \cdot p = 1 - p + pe^{t}$.

2. **Q**: How is the Bernoulli distribution related to the Binomial distribution?
   **A**: The sum of $n$ independent and identically distributed $\text{Bernoulli}(p)$ random variables follows a $\text{Binomial}(n, p)$ distribution.

3. **Q**: Show that the Bernoulli distribution belongs to the exponential family.
   **A**: Write $f(x; p) = p^x (1-p)^{1-x} = \exp\left[x \log(p) + (1-x)\log(1-p)\right] = \exp\left[x \log\left(\frac{p}{1-p}\right) + \log(1-p)\right]$. The natural parameter is $\theta = \log(p/(1-p))$, the sufficient statistic is $T(x) = x$, and $\log(1-p) = -\log(1+e^\theta)$.

4. **Q**: What is the conjugate prior for the Bernoulli distribution and why is it useful?
   **A**: The Beta distribution $\text{Beta}(\alpha, \beta)$ is the conjugate prior. This means the posterior is also Beta, making Bayesian inference computationally tractable. The posterior is $\text{Beta}(\alpha + \sum x_i, \beta + n - \sum x_i)$.

5. **Q**: In logistic regression, how does the Bernoulli distribution appear in the loss function?
   **A**: Logistic regression assumes each label $y_i \mid x_i \sim \text{Bernoulli}(p_i)$ where $p_i = \sigma(\beta^T x_i)$. The negative log-likelihood gives binary cross-entropy loss: $-\sum [y_i \log p_i + (1-y_i)\log(1-p_i)]$.

### Advanced

1. **Q**: Prove that the Bernoulli distribution has maximum entropy among all distributions on $\{0, 1\}$ with a given mean $p$.
   **A**: For a distribution $P$ on $\{0, 1\}$ with $P(1) = p$, the entropy is $H(P) = -p\log p - (1-p)\log(1-p)$. Any other distribution on $\{0,1\}$ with the same mean must assign the same probabilities to 0 and 1, so the Bernoulli distribution is the unique distribution on this support. The maximum entropy property holds more generally for the Bernoulli among all distributions with support $\{0,1\}$ and mean $p$.

2. **Q**: Derive the Fisher information for the parameter $p$ of a Bernoulli distribution. How does it relate to the variance of the MLE?
   **A**: The Fisher information for a single observation is $\mathcal{I}(p) = \mathbb{E}\left[\left(\frac{\partial \log f}{\partial p}\right)^2\right] = \mathbb{E}\left[\left(\frac{X}{p} - \frac{1-X}{1-p}\right)^2\right] = \frac{1}{p(1-p)}$. For $n$ observations, $\mathcal{I}_n(p) = \frac{n}{p(1-p)}$. The asymptotic variance of the MLE $\hat{p}$ is $1/\mathcal{I}_n(p) = p(1-p)/n$.

3. **Q**: In the context of Bayesian binary classification with a logistic regression model, derive the posterior distribution of the weights $w$ using a Gaussian prior and a Bernoulli likelihood. Why is exact inference intractable?
   **A**: The prior is $p(w) = \mathcal{N}(0, \Sigma)$. The likelihood is $p(y \mid X, w) = \prod_{i=1}^n \sigma(w^T x_i)^{y_i} (1-\sigma(w^T x_i))^{1-y_i}$. The posterior is $p(w \mid X, y) \propto p(w) \cdot p(y \mid X, w)$. This posterior does not have a closed form because the sigmoid function $\sigma$ makes the product non-Gaussian. Exact inference is intractable, so we use approximations like Laplace approximation, variational inference, or MCMC.

## Practice Problems

### Easy

1. A biased coin has $p = 0.6$ for heads. Let $X = 1$ for heads, $X = 0$ for tails. Find $P(X = 1)$, $P(X = 0)$, $\mathbb{E}[X]$, and $\text{Var}[X]$.

2. A spam filter correctly identifies 95% of spam emails. For a single spam email, let $X = 1$ if correctly flagged. Find the distribution of $X$, its mean, and its variance.

3. If $X \sim \text{Bernoulli}(0.25)$, compute $P(X = 1) + P(X = 0)$.

4. For $X \sim \text{Bernoulli}(p)$, show that $\mathbb{E}[X^2] = p$.

5. A factory produces items with a 3% defect rate. For one randomly selected item, let $X = 1$ if defective. Find $P(X = 1)$ and $P(X = 0)$.

### Medium

1. If $X_1, X_2, \ldots, X_{10}$ are independent $\text{Bernoulli}(0.4)$ variables, find the distribution of $Y = \sum_{i=1}^{10} X_i$ and compute $P(Y = 3)$.

2. Let $X \sim \text{Bernoulli}(p)$. Find the value of $p$ that minimizes $\text{Var}[X]$. What is the minimum variance?

3. For $X \sim \text{Bernoulli}(p)$, compute the skewness $\gamma_1 = \mathbb{E}\left[\left(\frac{X-p}{\sqrt{p(1-p)}}\right)^3\right]$ in terms of $p$.

4. In a study, 8 out of 12 patients recovered after a treatment. Assuming each patient's recovery is $\text{Bernoulli}(p)$ independent of others, find the MLE of $p$ and its standard error.

5. Let $X \sim \text{Bernoulli}(0.2)$ and $Y \sim \text{Bernoulli}(0.6)$ be independent. Find $P(X + Y = 1)$.

### Hard

1. Suppose $X_1, \ldots, X_n$ are i.i.d. $\text{Bernoulli}(p)$. The sample variance is $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$. Find $\mathbb{E}[S^2]$ and show that it is an unbiased estimator of $\text{Var}[X_i]$.

2. Let $X \sim \text{Bernoulli}(p)$. The log-odds is $\theta = \log(p/(1-p))$. Express the PMF in terms of $\theta$ and show that the Bernoulli distribution is a member of the exponential family with natural parameter $\theta$.

3. A Bayesian statistician has a prior $p \sim \text{Beta}(1, 1)$ (uniform on $[0,1]$). After observing $n$ independent Bernoulli trials with $k$ successes, find the posterior distribution, the posterior mean, and the posterior mode. How do these compare to the MLE?

## Solutions

### Easy Solutions

1. $P(X = 1) = 0.6$, $P(X = 0) = 0.4$, $\mathbb{E}[X] = 0.6$, $\text{Var}[X] = 0.6 \times 0.4 = 0.24$.

2. $X \sim \text{Bernoulli}(0.95)$, $\mathbb{E}[X] = 0.95$, $\text{Var}[X] = 0.95 \times 0.05 = 0.0475$.

3. $P(X = 1) + P(X = 0) = p + (1-p) = 1$ always.

4. $\mathbb{E}[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p$.

5. $P(X = 1) = 0.03$, $P(X = 0) = 0.97$.

### Medium Solutions

1. $Y \sim \text{Binomial}(10, 0.4)$. $P(Y = 3) = \binom{10}{3}(0.4)^3(0.6)^7 \approx 0.2150$.

2. $\text{Var}[X] = p(1-p)$ is minimized at $p = 0$ or $p = 1$, giving $\text{Var}[X] = 0$.

3. Using the definition, $\gamma_1 = \frac{1-2p}{\sqrt{p(1-p)}}$.

4. MLE $\hat{p} = 8/12 = 2/3 \approx 0.6667$. $\text{SE}(\hat{p}) = \sqrt{(2/3)(1/3)/12} \approx 0.1361$.

5. $P(X + Y = 1) = P(X=1, Y=0) + P(X=0, Y=1) = 0.2 \times 0.4 + 0.8 \times 0.6 = 0.08 + 0.48 = 0.56$.

### Hard Solutions

1. Note that $\mathbb{E}[S^2] = \sigma^2 = p(1-p)$. For Bernoulli variables, $\bar{X} = \sum X_i / n$ is the sample proportion. The sample variance $S^2 = \frac{n}{n-1}\bar{X}(1-\bar{X})$ is unbiased because $\mathbb{E}[\bar{X}(1-\bar{X})] = \frac{n-1}{n}p(1-p)$.

2. Let $\theta = \log(p/(1-p))$. Then $p = \frac{e^\theta}{1+e^\theta}$ and $1-p = \frac{1}{1+e^\theta}$. The PMF is $f(x; \theta) = \left(\frac{e^\theta}{1+e^\theta}\right)^x \left(\frac{1}{1+e^\theta}\right)^{1-x} = e^{\theta x} \cdot \frac{1}{1+e^\theta} = \exp(\theta x - \log(1+e^\theta))$. This is in exponential family form with $T(x) = x$, $\theta$ as natural parameter, and $A(\theta) = \log(1+e^\theta)$.

3. Posterior: $p \mid \text{data} \sim \text{Beta}(1 + k, 1 + n - k)$. Posterior mean: $(1+k)/(n+2)$. Posterior mode: $k/n$ (same as MLE when $k$ and $n-k$ are positive). The posterior mean is shrunk toward $0.5$ compared to the MLE, which is a characteristic of Bayesian estimates with a uniform prior.

## Related Concepts

- **Binomial Distribution**: The sum of $n$ independent Bernoulli($p$) trials forms a Binomial($n, p$) distribution.
- **Geometric Distribution**: The number of Bernoulli trials needed to get the first success follows a Geometric distribution.
- **Negative Binomial Distribution**: The number of Bernoulli trials needed to get $r$ successes follows a Negative Binomial distribution.
- **Logistic Regression**: A generalized linear model where the response is Bernoulli and the link function is the logit.
- **Categorical Distribution**: The generalization of Bernoulli to more than two categories.
- **Beta Distribution**: The conjugate prior for the Bernoulli parameter $p$.
- **Binary Cross-Entropy**: The loss function derived from the Bernoulli likelihood used in binary classification.
- **Exponential Family**: The Bernoulli distribution is a member of the exponential family.

## Next Concepts

- Binomial Distribution (MATH-073)
- Multinomial Distribution
- Beta Distribution
- Logistic Regression
- Generalized Linear Models (GLMs)

## Summary

The Bernoulli distribution is the simplest discrete probability distribution, modelling a random experiment with exactly two outcomes: success ($X = 1$) with probability $p$, and failure ($X = 0$) with probability $1-p$. Its expected value is $p$ and its variance is $p(1-p)$. Despite its simplicity, it is of paramount importance in probability theory and machine learning. It forms the building block of the Binomial, Geometric, and Negative Binomial distributions. In machine learning, it underpins binary classification through logistic regression, and the binary cross-entropy loss function is derived from the Bernoulli likelihood. Understanding the Bernoulli distribution thoroughly is essential before studying more complex discrete distributions.

## Key Takeaways

- The Bernoulli distribution models any binary outcome with probability $p$ of success
- PMF: $P(X = x) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$
- Mean $\mathbb{E}[X] = p$, Variance $\text{Var}[X] = p(1-p)$
- Sum of $n$ i.i.d. Bernoulli($p$) variables is Binomial($n, p$)
- Binary cross-entropy loss in logistic regression is the negative log-likelihood of the Bernoulli distribution
- MLE of $p$ is the sample proportion of successes
- The Bernoulli distribution belongs to the exponential family with natural parameter $\theta = \log(p/(1-p))$
- The Beta distribution is the conjugate prior for $p$
