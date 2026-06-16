# Concept: Poisson Distribution

## Concept ID

MATH-074

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Domain

Mathematics

## Module

Probability

## Learning Objectives

1. Define the Poisson distribution and its probability mass function
2. Compute probabilities, mean, variance, and cumulative probabilities for Poisson random variables
3. Understand the Poisson distribution as a limit of the Binomial distribution for rare events
4. Apply the Poisson distribution to model count data in real-world scenarios
5. Connect the Poisson distribution to Poisson regression, rare event modelling, and count data analysis in machine learning

## Prerequisites

- Binomial distribution (MATH-073)
- Limit of a sequence
- Taylor series expansion of $e^x$
- Expected value and variance of a discrete random variable
- Concept of rate parameters

## Definition

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space, given that these events occur with a known constant mean rate $\lambda > 0$ and independently of the time since the last event. A random variable $X$ following a Poisson distribution is denoted as

$$X \sim \text{Poisson}(\lambda)$$

The probability mass function (PMF) is given by

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

## Intuition

The Poisson distribution models the count of rare events that happen randomly over time or space. The classic example is the number of emails received in an hour. If you receive an average of 10 emails per hour, then the actual count in any given hour follows a Poisson distribution with $\lambda = 10$.

The key intuition is that the Poisson distribution arises from a process where events occur continuously and independently at a constant average rate. This is called a Poisson process. The distribution tells us how likely it is to observe 0, 1, 2, ... events in a fixed interval.

Think of the Poisson as a "counting distribution" for rare events. It differs from the Binomial in that there is no fixed number of trials $n$; instead, events can occur any number of times (theoretically unbounded), though the probability of very large counts becomes vanishingly small.

## Why This Concept Matters

The Poisson distribution is essential for modelling count data across virtually all scientific disciplines:

- It is the standard model for count data where events occur randomly over time or space
- It provides the theoretical foundation for Poisson regression (a type of generalized linear model)
- It models rare events in epidemiology (disease counts), finance (market crashes), and natural disasters
- It describes queueing processes (customer arrivals, network traffic)
- It is used in insurance for claims modelling and in reliability engineering for failure rates
- The Poisson process is fundamental to stochastic process theory

## Historical Background

The distribution is named after Siméon Denis Poisson (1781--1840), a French mathematician and physicist. Poisson published the distribution in 1837 in his work *Recherches sur la Probabilité des Jugements en Matière Criminelle et en Matière Civile* (Research on the Probability of Judgments in Criminal and Civil Matters). Interestingly, he derived the distribution as a limiting case of the Binomial distribution. The distribution gained prominence later through the work of Ladislaus Bortkiewicz, who in 1898 used it to model the number of Prussian soldiers killed by horse kicks per year — a classic early example of Poisson modelling. The Poisson process was formally developed in the early 20th century, and the distribution became a cornerstone of stochastic process theory.

## Real World Examples

1. **Call Center**: A call centre receives an average of 40 calls per hour. The number of calls received in a randomly chosen hour follows $X \sim \text{Poisson}(40)$.

2. **Website Traffic**: A website gets an average of 150 visits per hour. The number of visits in the next hour follows $X \sim \text{Poisson}(150)$.

3. **Radioactive Decay**: A radioactive sample emits an average of 3 alpha particles per second. The count in one second follows $X \sim \text{Poisson}(3)$.

4. **Insurance Claims**: An insurance company receives an average of 5 claims per day for a particular policy type. The daily claim count follows $X \sim \text{Poisson}(5)$.

5. **Epidemiology**: A rare disease occurs at a rate of 2 cases per 100,000 people per year. In a city of 500,000, the annual number of cases follows $X \sim \text{Poisson}(10)$.

6. **Text Mining**: The number of times a specific word appears in a document of fixed length often follows a Poisson distribution (or a related distribution like the Negative Binomial when overdispersed).

## AI/ML Relevance

The Poisson distribution appears in several areas of machine learning and data science.

**Poisson Regression**: This is a generalized linear model (GLM) for count data. The response variable $Y$ is assumed to follow a Poisson distribution, and the log of its mean is modelled as a linear combination of predictors:

$$\log(\mathbb{E}[Y \mid X]) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

Poisson regression is used for predicting counts, such as the number of customer purchases, number of website clicks, or number of insurance claims. It is the natural counterpart to logistic regression (which handles binary outcomes) for count outcomes.

**Rare Event Modelling**: In fraud detection, fraudulent transactions are rare events (e.g., 0.1% of all transactions). The count of fraudulent transactions per day can be modelled with a Poisson distribution. Poisson regression helps identify factors associated with higher fraud rates.

**Queueing Theory for System Design**: In recommendation systems and web services, the arrival rate of user requests often follows a Poisson process. This informs capacity planning, load balancing, and latency optimization.

**Natural Language Processing**: The Poisson distribution is used in topic modelling. In Latent Dirichlet Allocation (LDA), the number of words in a document can be modelled with a Poisson distribution. Some neural topic models also use Poisson assumptions about word counts.

**Network Traffic Modelling**: The number of packets arriving at a network node per unit time is often modelled as a Poisson process, which helps in designing congestion control algorithms and predicting network load.

**Spatial Point Processes**: In computer vision and geospatial analysis, the Poisson process models the random distribution of points in space (e.g., locations of trees in a forest, defects on a wafer). Ripley's K-function and other spatial statistics rely on the Poisson assumption.

**Zero-Inflated Models**: In many real-world count datasets, there are more zeros than the Poisson predicts. Zero-inflated Poisson (ZIP) models and hurdle models extend the Poisson to handle excess zeros, which is common in ML applications like predicting the number of purchases (many customers buy nothing).

## Mathematical Explanation

### Probability Mass Function

For $X \sim \text{Poisson}(\lambda)$,

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

The PMF is defined for all non-negative integers. It sums to 1:

$$\sum_{k=0}^{\infty} \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = e^{-\lambda} e^{\lambda} = 1$$

### Derivation from the Binomial Distribution

The Poisson distribution can be derived as a limiting case of the Binomial distribution. Consider $X \sim \text{Binomial}(n, p)$ with $n \to \infty$ and $p \to 0$ such that $np = \lambda$ remains constant. Then:

$$\lim_{n \to \infty} P(X = k) = \lim_{n \to \infty} \binom{n}{k} p^k (1-p)^{n-k}$$

Substitute $p = \lambda/n$:

$$= \lim_{n \to \infty} \frac{n!}{k!(n-k)!} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

$$= \frac{\lambda^k}{k!} \lim_{n \to \infty} \frac{n!}{(n-k)! n^k} \left(1 - \frac{\lambda}{n}\right)^{n} \left(1 - \frac{\lambda}{n}\right)^{-k}$$

As $n \to \infty$, $\frac{n!}{(n-k)! n^k} \to 1$, $(1 - \lambda/n)^n \to e^{-\lambda}$, and $(1 - \lambda/n)^{-k} \to 1$. Hence:

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

### Cumulative Distribution Function

The CDF is:

$$F(k) = P(X \leq k) = \sum_{i=0}^k \frac{e^{-\lambda} \lambda^i}{i!}$$

This can be expressed in terms of the incomplete gamma function:

$$P(X \leq k) = \frac{\Gamma(k+1, \lambda)}{k!}$$

where $\Gamma(s, x)$ is the upper incomplete gamma function.

### Expected Value

$$\mathbb{E}[X] = \sum_{k=0}^{\infty} k \cdot \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^k}{(k-1)!} = \lambda e^{-\lambda} \sum_{j=0}^{\infty} \frac{\lambda^j}{j!} = \lambda e^{-\lambda} e^{\lambda} = \lambda$$

### Variance

$$\mathbb{E}[X^2] = \sum_{k=0}^{\infty} k^2 \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=1}^{\infty} k \frac{\lambda^k}{(k-1)!} = e^{-\lambda} \sum_{k=1}^{\infty} ((k-1)+1) \frac{\lambda^k}{(k-1)!}$$

$$= e^{-\lambda} \left( \sum_{k=2}^{\infty} \frac{\lambda^k}{(k-2)!} + \sum_{k=1}^{\infty} \frac{\lambda^k}{(k-1)!} \right)$$

$$= e^{-\lambda} (\lambda^2 e^{\lambda} + \lambda e^{\lambda}) = \lambda^2 + \lambda$$

Therefore:

$$\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = (\lambda^2 + \lambda) - \lambda^2 = \lambda$$

A remarkable property: the mean equals the variance.

### Moment-Generating Function

$$M_X(t) = \mathbb{E}[e^{tX}] = \sum_{k=0}^{\infty} e^{tk} \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} e^{\lambda e^t} = \exp(\lambda(e^t - 1))$$

### Skewness and Kurtosis

Skewness: $\gamma_1 = \frac{1}{\sqrt{\lambda}}$

Excess kurtosis: $\gamma_2 = \frac{1}{\lambda}$

As $\lambda$ increases, the distribution becomes more symmetric and more Normal-like.

## Formula(s)

1. **PMF**: $P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$

2. **Mean**: $\mathbb{E}[X] = \lambda$

3. **Variance**: $\text{Var}[X] = \lambda$

4. **Standard Deviation**: $\sigma_X = \sqrt{\lambda}$

5. **MGF**: $M_X(t) = \exp(\lambda(e^t - 1))$

6. **CDF**: $F(k) = \sum_{i=0}^k \frac{e^{-\lambda} \lambda^i}{i!}$

## Properties

1. **Equidispersion**: The mean equals the variance, $\mathbb{E}[X] = \text{Var}[X] = \lambda$. This is a defining characteristic of the Poisson distribution. Real data often exhibit overdispersion (variance > mean), which motivates the Negative Binomial distribution.

2. **Sum of independent Poissons**: If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ are independent, then $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$.

3. **Reproductivity**: The Poisson family is reproductive under addition. This follows from property 2.

4. **Normal approximation**: For large $\lambda$, $\text{Poisson}(\lambda) \approx \mathcal{N}(\lambda, \lambda)$. The approximation is good when $\lambda \geq 20$.

5. **Poisson process**: The Poisson distribution arises from a Poisson process with rate $\lambda$, which has the following properties:
   - The number of events in disjoint time intervals are independent
   - The number of events in an interval of length $t$ follows $\text{Poisson}(\lambda t)$
   - The inter-arrival times follow $\text{Exponential}(\lambda)$

6. **Exponential family**: The Poisson distribution belongs to the exponential family with natural parameter $\theta = \log \lambda$.

7. **Thinning**: If $X \sim \text{Poisson}(\lambda)$ and each event is independently classified as type A with probability $p$ and type B with probability $1-p$, then the counts of type A and type B events are independent Poisson variables with means $\lambda p$ and $\lambda(1-p)$ respectively.

8. **Conjugate prior**: The Gamma distribution is the conjugate prior for $\lambda$ in the Poisson model.

## Step-by-Step Worked Examples

### Example 1: Call Centre Calls

A call centre receives an average of 12 calls per hour. What is the probability of receiving exactly 8 calls in the next hour? What about at most 5 calls?

**Step 1**: Identify the parameter. $\lambda = 12$ calls per hour.

**Step 2**: Distribution. $X \sim \text{Poisson}(12)$.

**Step 3**: Probability of exactly 8 calls.

$$P(X = 8) = \frac{e^{-12} \cdot 12^8}{8!}$$

Compute step by step:
- $12^8 = 429,981,696$
- $8! = 40,320$
- $12^8 / 8! = 429,981,696 / 40,320 = 10,664$
- $e^{-12} \approx 6.144 \times 10^{-6}$

$$P(X = 8) \approx 6.144 \times 10^{-6} \times 10,664 \approx 0.0655$$

**Step 4**: Probability of at most 5 calls.

$$P(X \leq 5) = \sum_{k=0}^{5} \frac{e^{-12} \cdot 12^k}{k!}$$

$$P(X = 0) = e^{-12} \approx 6.14 \times 10^{-6}$$
$$P(X = 1) = e^{-12} \cdot 12 \approx 7.37 \times 10^{-5}$$
$$P(X = 2) = e^{-12} \cdot 144/2 \approx 4.42 \times 10^{-4}$$
$$P(X = 3) = e^{-12} \cdot 1728/6 \approx 0.00177$$
$$P(X = 4) = e^{-12} \cdot 20736/24 \approx 0.00531$$
$$P(X = 5) = e^{-12} \cdot 248832/120 \approx 0.01274$$

$$P(X \leq 5) \approx 0.00000614 + 0.0000737 + 0.000442 + 0.00177 + 0.00531 + 0.01274 \approx 0.02034$$

**Step 5**: Interpret. There is about a 6.55% chance of exactly 8 calls and about a 2.03% chance of at most 5 calls.

### Example 2: Rare Disease

A rare disease occurs at a rate of 1 case per 10,000 people per year. In a city of 50,000 people, what is the probability of observing 0 cases in a year? What about 3 or more cases?

**Step 1**: Compute $\lambda$. $\lambda = 50,000 \times (1/10,000) = 5$ cases per year.

**Step 2**: Distribution. $X \sim \text{Poisson}(5)$.

**Step 3**: Probability of 0 cases.

$$P(X = 0) = \frac{e^{-5} \cdot 5^0}{0!} = e^{-5} \approx 0.00674$$

**Step 4**: Probability of 3 or more cases. Use the complement.

$$P(X \geq 3) = 1 - P(X \leq 2) = 1 - [P(X = 0) + P(X = 1) + P(X = 2)]$$

$$P(X = 1) = e^{-5} \cdot 5 \approx 0.03369$$
$$P(X = 2) = e^{-5} \cdot 25/2 \approx 0.08422$$

$$P(X \geq 3) = 1 - (0.00674 + 0.03369 + 0.08422) = 1 - 0.12465 = 0.87535$$

**Step 5**: Interpret. There is a 0.67% chance of no cases and an 87.5% chance of 3 or more cases.

### Example 3: Poisson Regression

A data scientist wants to model the number of product purchases per customer based on the number of website visits. The Poisson regression model gives:

$$\log(\text{purchases}) = 0.5 + 0.3 \times \text{visits}$$

For a customer who visits 4 times, predict the expected number of purchases and find the probability of exactly 2 purchases.

**Step 1**: Compute the predicted $\lambda$.

$$\log(\lambda) = 0.5 + 0.3 \times 4 = 0.5 + 1.2 = 1.7$$
$$\lambda = e^{1.7} \approx 5.474$$

**Step 2**: Distribution. $X \mid (\text{visits} = 4) \sim \text{Poisson}(5.474)$.

**Step 3**: Expected number of purchases.

$$\mathbb{E}[X] = \lambda = 5.474$$

**Step 4**: Probability of exactly 2 purchases.

$$P(X = 2) = \frac{e^{-5.474} \cdot 5.474^2}{2!} = \frac{e^{-5.474} \cdot 29.96}{2} \approx \frac{0.00421 \times 29.96}{2} \approx 0.0631$$

**Step 5**: Interpret. A customer with 4 visits is expected to make about 5.47 purchases. The probability of making exactly 2 purchases is about 6.31%.

### Example 4: Normal Approximation

Suppose the number of daily website visits follows $\text{Poisson}(200)$. Use the normal approximation to find the probability that the number of visits exceeds 220 on a given day.

**Step 1**: Parameters. $\lambda = 200$.

**Step 2**: Normal approximation. $X \approx \mathcal{N}(200, 200)$.

**Step 3**: Standard deviation. $\sigma = \sqrt{200} \approx 14.142$.

**Step 4**: Compute z-score for $X = 220$ (with continuity correction, use $X \geq 220.5$).

$$z = \frac{220.5 - 200}{14.142} \approx 1.449$$

**Step 5**: Compute probability.

$$P(X > 220) \approx 1 - \Phi(1.449) \approx 1 - 0.9265 = 0.0735$$

**Step 6**: Interpret. There is about a 7.35% chance that daily visits exceed 220.

### Example 5: Thinning Property

A text message gateway processes messages at an average rate of 100 per minute. 60% are SMS and 40% are MMS. What is the distribution of the number of SMS messages per minute? What is the probability of exactly 70 SMS messages?

**Step 1**: Total rate. $\lambda = 100$ per minute.

**Step 2**: Thinning property. SMS count $X_S \sim \text{Poisson}(100 \times 0.6) = \text{Poisson}(60)$.

**Step 3**: Probability of exactly 70 SMS messages.

$$P(X_S = 70) = \frac{e^{-60} \cdot 60^{70}}{70!}$$

This is computationally intensive. Use the normal approximation with $\lambda = 60$.

$$z = \frac{70.5 - 60}{\sqrt{60}} \approx \frac{10.5}{7.746} \approx 1.355$$

$$P(X_S = 70) \approx \frac{1}{\sqrt{2\pi \times 60}} \exp\left(-\frac{(70-60)^2}{2 \times 60}\right) \approx 0.051$$

Or from Poisson tables/software: $P(X_S = 70) \approx 0.0456$.

## Visual Interpretation

The Poisson distribution can be visualized as a bar chart with bars at $k = 0, 1, 2, \ldots$ where the height of each bar is $P(X = k) = e^{-\lambda} \lambda^k / k!$.

**Shape depends on $\lambda$**:
- For small $\lambda$ ($\lambda < 1$), the distribution is highly skewed right. The bar at $k = 0$ is the tallest, and probabilities decrease rapidly.
- For moderate $\lambda$ ($1 \leq \lambda \leq 10$), the distribution is skewed right but becomes more bell-shaped.
- For large $\lambda$ ($\lambda > 20$), the distribution is approximately symmetric and bell-shaped, resembling the Normal distribution.

**Key visual feature**: As $\lambda$ increases, the distribution shifts to the right, becomes more spread out (since variance equals $\lambda$), and becomes more symmetric.

The Poisson distribution is always unimodal, with the mode at $\lfloor \lambda \rfloor$ or $\lfloor \lambda \rfloor - 1$.

## Common Mistakes

1. **Assuming independence when it does not hold**: The Poisson distribution assumes events occur independently. If events are clustered (e.g., contagious diseases), the true distribution is overdispersed relative to Poisson, and the Negative Binomial distribution may be more appropriate.

2. **Confusing rate $\lambda$ with count**: $\lambda$ is the average rate per interval, not the maximum possible count. The Poisson distribution has no upper bound, though probabilities become negligible for large $k$.

3. **Using Poisson when the Binomial is appropriate**: If there is a fixed maximum number of trials $n$, the Binomial distribution is correct, not Poisson. The Poisson is a limiting approximation to the Binomial when $n$ is large and $p$ is small.

4. **Ignoring the equal mean-variance property**: If the sample variance is much larger than the sample mean (overdispersion) or much smaller (underdispersion), the Poisson model may be inappropriate. This is a key diagnostic in Poisson regression.

5. **Applying Poisson to non-count data**: The Poisson distribution is for count data (non-negative integers). It should not be used for continuous data, proportions, or categorical data.

6. **Misinterpreting $\lambda$ as a probability**: $\lambda$ is a rate (average count), not a probability. It can be any positive real number, not just between 0 and 1.

7. **Forgetting the factorial in the denominator**: The $k!$ term is essential. Omitting it leads to incorrect calculations.

8.**Overlooking the Poisson process assumptions**: The Poisson distribution from a Poisson process assumes events occur at a constant rate and independently. If the rate changes over time, use a non-homogeneous Poisson process.

## Interview Questions

### Beginner

1. **Q**: What are the possible values of a Poisson random variable?
   **A**: All non-negative integers: $k = 0, 1, 2, \ldots$ (unbounded).

2. **Q**: What is the mean and variance of a $\text{Poisson}(\lambda)$ random variable?
   **A**: Both the mean and variance equal $\lambda$.

3. **Q**: If $X \sim \text{Poisson}(3)$, compute $P(X = 0)$.
   **A**: $P(X = 0) = e^{-3} \cdot 3^0 / 0! = e^{-3} \approx 0.0498$.

4. **Q**: What does the parameter $\lambda$ represent in a Poisson distribution?
   **A**: $\lambda$ represents the average rate (expected number of events) per unit time or space.

5. **Q**: Name a real-world example where the Poisson distribution might apply.
   **A**: The number of customers arriving at a store per hour, the number of typos per page in a book, the number of earthquakes per year in a region, etc.

### Intermediate

1. **Q**: Derive the Poisson distribution as a limiting case of the Binomial distribution.
   **A**: Take $\text{Binomial}(n, p)$ with $n \to \infty$, $p \to 0$, $np = \lambda$ constant. Then $\lim_{n\to\infty} \binom{n}{k} p^k (1-p)^{n-k} = e^{-\lambda} \lambda^k / k!$. The derivation uses the approximation $\binom{n}{k} \approx n^k/k!$, $(1-p)^{n-k} \approx e^{-\lambda}$, and $p^k = (\lambda/n)^k$.

2. **Q**: What is the moment-generating function of the Poisson distribution?
   **A**: $M_X(t) = \exp(\lambda(e^t - 1))$.

3. **Q**: If $X \sim \text{Poisson}(3)$ and $Y \sim \text{Poisson}(5)$ are independent, what is the distribution of $X + Y$?
   **A**: $X + Y \sim \text{Poisson}(8)$ (sum of independent Poissons is Poisson with sum of rates).

4. **Q**: What is overdispersion and why is it a problem for Poisson models?
   **A**: Overdispersion occurs when the variance exceeds the mean (a violation of the Poisson assumption). It leads to underestimated standard errors, narrow confidence intervals, and inflated test statistics. Negative Binomial regression or quasi-Poisson models can address this.

5. **Q**: Explain the thinning property of the Poisson distribution.
   **A**: If $X \sim \text{Poisson}(\lambda)$ and each event is independently classified as type A with probability $p$, then the count of type A events $X_A \sim \text{Poisson}(\lambda p)$ and is independent of the count of type B events.

### Advanced

1. **Q**: Show that the Poisson distribution belongs to the exponential family. Identify the natural parameter, sufficient statistic, and cumulant function.
   **A**: $f(k; \lambda) = \frac{e^{-\lambda} \lambda^k}{k!} = \exp(k \log \lambda - \lambda - \log k!)$. The natural parameter is $\theta = \log \lambda$. The sufficient statistic is $T(k) = k$. The cumulant function is $A(\theta) = e^\theta = \lambda$. The base measure is $h(k) = 1/k!$.

2. **Q**: Derive the Fisher information for $\lambda$ in a Poisson model. What is the Cramér-Rao lower bound for an unbiased estimator of $\lambda$?
   **A**: The log-likelihood is $\ell(\lambda) = -\lambda + k\log \lambda - \log k!$. The score function is $\partial \ell/\partial \lambda = -1 + k/\lambda$. The Fisher information is $\mathcal{I}(\lambda) = \mathbb{E}[(-1 + k/\lambda)^2] = \text{Var}(k)/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$. For $n$ observations, $\mathcal{I}_n(\lambda) = n/\lambda$. The CRLB for an unbiased estimator is $\lambda/n$, and $\hat{\lambda} = \bar{X}$ achieves this bound.

3. **Q**: The Poisson distribution is often used for count data, but real data frequently exhibit zero-inflation. Describe two modelling approaches that handle excess zeros and how they modify the Poisson likelihood.
   **A**: (1) Zero-Inflated Poisson (ZIP): Models the data as a mixture: with probability $\pi$, the count is 0 from a "structural zero" process, and with probability $1-\pi$, the count follows $\text{Poisson}(\lambda)$. The PMF is $P(X=0) = \pi + (1-\pi)e^{-\lambda}$, $P(X=k) = (1-\pi) e^{-\lambda} \lambda^k/k!$ for $k \geq 1$. (2) Hurdle model: A Bernoulli process determines whether the count is zero or positive, and a truncated Poisson models the positive counts. The PMF is $P(X=0) = \pi$, $P(X=k) = (1-\pi) \frac{e^{-\lambda} \lambda^k}{k!(1-e^{-\lambda})}$ for $k \geq 1$.

## Practice Problems

### Easy

1. If $X \sim \text{Poisson}(4)$, compute $P(X = 2)$.

2. A bookstore receives an average of 3 customers per hour. What is the probability of exactly 5 customers in the next hour?

3. If $X \sim \text{Poisson}(0.5)$, find $P(X = 0)$ and $P(X \geq 1)$.

4. The number of typos per page in a book follows $\text{Poisson}(0.2)$. Find the probability that a randomly selected page has no typos.

5. For $X \sim \text{Poisson}(10)$, compute $\mathbb{E}[X]$ and $\text{Var}[X]$.

### Medium

1. The number of traffic accidents at an intersection per month follows $\text{Poisson}(2.5)$. Find the probability of at least 4 accidents in a month.

2. Show that if $X \sim \text{Poisson}(\lambda)$, then $\mathbb{E}[X(X-1)] = \lambda^2$. Use this to derive $\text{Var}[X]$.

3. A hospital emergency room receives an average of 8 patients per hour. Using the normal approximation, find the probability of receiving more than 12 patients in a given hour.

4. If $X \sim \text{Poisson}(\lambda)$ and $Y \sim \text{Poisson}(2\lambda)$ are independent, find $P(X + Y = 3)$ in terms of $\lambda$.

5. For a Poisson distribution, show that the skewness is $1/\sqrt{\lambda}$ and the excess kurtosis is $1/\lambda$.

### Hard

1. Let $X_1, X_2, \ldots, X_n$ be i.i.d. $\text{Poisson}(\lambda)$. Derive the MLE of $\lambda$, its asymptotic distribution, and construct an approximate 95% confidence interval for $\lambda$.

2. The number of emails received per hour follows $\text{Poisson}(\lambda)$. In a 5-hour workday, the counts are 3, 7, 5, 4, 6. Find the MLE of $\lambda$, compute the standard error, and test $H_0: \lambda = 4$ against $H_1: \lambda \neq 4$ at $\alpha = 0.05$.

3. In a Poisson regression model, the deviance residual is defined as $r_D = \text{sign}(y - \hat{\mu})\sqrt{2\left[y\log\frac{y}{\hat{\mu}} - (y - \hat{\mu})\right]}$. Derive this expression from the difference in log-likelihoods between the saturated model and the fitted model.

## Solutions

### Easy Solutions

1. $P(X = 2) = e^{-4} \cdot 4^2/2! = e^{-4} \cdot 16/2 = 8e^{-4} \approx 0.1465$.

2. $P(X = 5) = e^{-3} \cdot 3^5/5! = e^{-3} \cdot 243/120 = 2.025e^{-3} \approx 0.1008$.

3. $P(X = 0) = e^{-0.5} \approx 0.6065$. $P(X \geq 1) = 1 - P(X = 0) = 1 - 0.6065 = 0.3935$.

4. $P(X = 0) = e^{-0.2} \approx 0.8187$.

5. $\mathbb{E}[X] = \text{Var}[X] = 10$.

### Medium Solutions

1. $P(X \geq 4) = 1 - P(X \leq 3)$. $P(X=0) = e^{-2.5} \approx 0.0821$, $P(X=1) = e^{-2.5} \cdot 2.5 \approx 0.2052$, $P(X=2) = e^{-2.5} \cdot 6.25/2 \approx 0.2565$, $P(X=3) = e^{-2.5} \cdot 15.625/6 \approx 0.2138$. $P(X \leq 3) \approx 0.7576$. $P(X \geq 4) \approx 0.2424$.

2. $\mathbb{E}[X(X-1)] = \sum_{k=0}^\infty k(k-1)e^{-\lambda}\lambda^k/k! = \lambda^2 e^{-\lambda} \sum_{k=2}^\infty \lambda^{k-2}/(k-2)! = \lambda^2 e^{-\lambda} e^{\lambda} = \lambda^2$. Then $\mathbb{E}[X^2] = \mathbb{E}[X(X-1)] + \mathbb{E}[X] = \lambda^2 + \lambda$, so $\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = (\lambda^2 + \lambda) - \lambda^2 = \lambda$.

3. $\lambda = 8$. Normal approx: $X \approx \mathcal{N}(8, 8)$. $z = (12.5 - 8)/\sqrt{8} \approx 4.5/2.828 \approx 1.591$. $P(X > 12) \approx 1 - \Phi(1.591) \approx 0.0559$.

4. $X + Y \sim \text{Poisson}(3\lambda)$. $P(X+Y=3) = e^{-3\lambda} (3\lambda)^3 / 3! = e^{-3\lambda} \cdot 27\lambda^3 / 6 = (9/2) \lambda^3 e^{-3\lambda}$.

5. Third central moment: $\mathbb{E}[(X-\lambda)^3] = \lambda$. Fourth central moment: $\mathbb{E}[(X-\lambda)^4] = 3\lambda^2 + \lambda$. Skewness $= \lambda / \lambda^{3/2} = 1/\sqrt{\lambda}$. Excess kurtosis $= (3\lambda^2 + \lambda)/\lambda^2 - 3 = 3 + 1/\lambda - 3 = 1/\lambda$.

### Hard Solutions

1. Log-likelihood: $\ell(\lambda) = \sum_{i=1}^n (-\lambda + x_i \log \lambda - \log x_i!)$. Score: $\partial \ell/\partial \lambda = -n + \sum x_i / \lambda = 0 \Rightarrow \hat{\lambda} = \bar{X}$. Fisher information: $\mathcal{I}(\lambda) = n/\lambda$. Asymptotic distribution: $\hat{\lambda} \approx \mathcal{N}(\lambda, \lambda/n)$. 95% CI: $\hat{\lambda} \pm 1.96 \sqrt{\hat{\lambda}/n}$.

2. $\hat{\lambda} = (3+7+5+4+6)/5 = 25/5 = 5$. $\text{SE}(\hat{\lambda}) = \sqrt{5/5} = 1$. Test statistic: $z = (5-4)/1 = 1$. Two-tailed $p$-value $= 2 \times (1 - \Phi(1)) \approx 0.3174$. Fail to reject $H_0$ at $\alpha = 0.05$.

3. For a single observation $y$ with fitted mean $\hat{\mu}$, the saturated model gives $\hat{\mu}_s = y$ (perfect fit). The contribution to deviance is $2[\ell(y; y) - \ell(\hat{\mu}; y)] = 2[y\log y - y - y\log\hat{\mu} + \hat{\mu}] = 2[y\log(y/\hat{\mu}) - (y - \hat{\mu})]$. The deviance residual is the signed square root, where the sign indicates whether the observation is above or below the fitted mean.

## Related Concepts

- **Binomial Distribution (MATH-073)**: The Poisson is the limiting case of the Binomial for rare events.
- **Normal Distribution (MATH-075)**: The Poisson approximates the Normal for large $\lambda$.
- **Central Limit Theorem (MATH-076)**: The CLT justifies the normal approximation to the Poisson for large $\lambda$.
- **Exponential Distribution**: The inter-arrival times in a Poisson process follow the Exponential distribution.
- **Gamma Distribution**: The waiting time for $k$ events in a Poisson process follows the Gamma distribution.
- **Negative Binomial Distribution**: An overdispersed alternative to the Poisson that allows variance > mean.
- **Poisson Regression**: A GLM for count data with Poisson likelihood.
- **Zero-Inflated Poisson (ZIP)**: Extension for count data with excess zeros.
- **Poisson Process**: The stochastic process that generates Poisson counts.

## Next Concepts

- Normal Distribution (MATH-075)
- Central Limit Theorem (MATH-076)
- Exponential Distribution
- Poisson Process
- Generalized Linear Models (GLMs)
- Survival Analysis

## Summary

The Poisson distribution models the count of events occurring independently at a constant average rate $\lambda$ over a fixed interval of time or space. Its PMF is $P(X = k) = e^{-\lambda} \lambda^k/k!$ for $k = 0, 1, 2, \ldots$, and it has the unique property that the mean equals the variance ($\mathbb{E}[X] = \text{Var}[X] = \lambda$). It can be derived as a limiting case of the Binomial distribution when $n$ is large and $p$ is small. The Poisson distribution is the foundation of Poisson regression for count data, appears in queueing theory and stochastic processes, and is widely used in epidemiology, insurance, and reliability engineering. In machine learning, it is used in Poisson regression, topic modelling, rare event detection, and network traffic modelling.

## Key Takeaways

- The Poisson distribution models count data with a rate parameter $\lambda$
- PMF: $P(X = k) = e^{-\lambda} \lambda^k/k!$, $k = 0, 1, 2, \ldots$
- Mean = Variance = $\lambda$ (equidispersion property)
- Sum of independent Poisson variables is Poisson with sum of rates
- The Poisson is the limiting distribution of Binomial$(n, p)$ as $n \to \infty$, $p \to 0$, $np = \lambda$
- The Poisson approximates the Normal for large $\lambda$
- Poisson regression models $\log(\mathbb{E}[Y \mid X])$ as a linear function of predictors
- Overdispersion (variance > mean) requires alternatives like Negative Binomial regression
- The Gamma distribution is the conjugate prior for $\lambda$
- Used in rare event modelling, fraud detection, queueing theory, and count data analysis
