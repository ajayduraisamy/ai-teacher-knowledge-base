# Concept: Probability Distribution

## Concept ID

MATH-071

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- Define a probability distribution as a complete description of a random variable's probabilities
- Distinguish between discrete distributions (PMF) and continuous distributions (PDF)
- Identify and apply key discrete distributions: Bernoulli, Binomial, Poisson
- Identify and apply key continuous distributions: Uniform, Normal, Exponential
- Define and interpret the cumulative distribution function (CDF)
- Connect probability distributions to model outputs in machine learning

## Prerequisites

- Random variable concepts (MATH-070)
- Probability fundamentals (MATH-065)
- Basic calculus (summation and integration)
- Conditional probability (MATH-068)

## Definition

A **probability distribution** is a mathematical function that gives the probabilities of occurrence of different possible outcomes of a random variable. It completely characterises the random variable's behaviour.

For a **discrete random variable**, the distribution is described by the **probability mass function (PMF)**, which assigns a probability to each value:
p_X(x) = P(X = x)

For a **continuous random variable**, the distribution is described by the **probability density function (PDF)**, which gives the probability density at each point. Probabilities are obtained by integrating over intervals:
P(a <= X <= b) = integral_a^b f_X(x) dx

The **cumulative distribution function (CDF)** for both types is:
F_X(x) = P(X <= x)

## Intuition

A probability distribution is a complete specification of how probability is spread across the possible values of a random variable. Think of it as a histogram of an infinite sample: if you could draw infinitely many samples from the distribution, the relative frequencies would converge to the PMF or PDF.

For discrete distributions, probability is concentrated at specific points (like 0.5 chance of heads, 0.5 chance of tails). For continuous distributions, probability is spread smoothly over an interval (like the bell curve of heights).

## Why This Concept Matters

Probability distributions are the building blocks of statistical modelling and machine learning. Every data-generating process is assumed to follow some distribution. Every machine learning model outputs a distribution over predictions. Understanding distributions is essential for choosing appropriate models, interpreting outputs, and quantifying uncertainty. Specific distributions like the Normal, Binomial, and Poisson are the workhorses of statistics, data science, and AI.

## Historical Background

The concept of a probability distribution evolved over centuries. **Abraham de Moivre** (1667-1754) discovered the normal distribution as an approximation to the binomial distribution. **Pierre-Simon Laplace** (1749-1827) generalised this and proved the first central limit theorem. **Carl Friedrich Gauss** (1777-1855) used the normal distribution in astronomy and geodesy, leading to its common name "Gaussian distribution."

**Siméon Denis Poisson** (1781-1840) derived the Poisson distribution as a limit of the binomial. **Francis Galton** (1822-1911) and **Karl Pearson** (1857-1936) catalogued many distributions and developed methods for fitting them to data. The rigorous mathematical foundation of distributions was established by **Andrey Kolmogorov** and **Henri Lebesgue** in the early 20th century through measure theory.

## Real World Examples

1. **Quality Control (Binomial)** - The number of defective items in a batch of 100 follows a Binomial distribution. Manufacturers use this to set acceptable quality levels and design sampling plans.

2. **Call Centers (Poisson)** - The number of phone calls arriving per minute follows a Poisson distribution. Managers use this to determine staffing levels needed to keep wait times acceptable.

3. **Heights (Normal)** - Heights of adult humans approximately follow a Normal distribution. Clothing manufacturers use this to determine size distributions for mass-produced garments.

4. **Waiting Times (Exponential)** - The time until the next bus arrives at a stop follows an Exponential distribution if arrivals are memoryless. Transit planners use this to model and improve service reliability.

5. **Random Number Generation (Uniform)** - Computers generate Uniform(0,1) random numbers as the basis for all other random sampling. This is the foundation of Monte Carlo simulation.

## AI/ML Relevance

1. **Softmax as Categorical Distribution** - The softmax function converts real-valued logits into a categorical probability distribution over classes. For K classes, softmax(z)_i = exp(z_i) / sum_j exp(z_j), producing a valid PMF that sums to 1. This is the output layer for most multi-class classifiers.

2. **Gaussian Processes** - A Gaussian process defines a distribution over functions. At any finite set of input points, the function values follow a multivariate normal distribution. GPs are used for regression, optimisation, and as priors over functions in Bayesian inference.

3. **Output Distributions of Models** - Neural networks with a final softmax layer output a categorical distribution. Regression models with Gaussian output layers output a normal distribution with predicted mean and variance. Variational autoencoders output distribution parameters (mean and variance of latent codes).

4. **Loss Functions from Distributions** - Common loss functions correspond to negative log-likelihood under specific distributions. Mean squared error corresponds to a Gaussian likelihood. Cross-entropy corresponds to a categorical likelihood. Binary cross-entropy corresponds to a Bernoulli likelihood.

5. **Mixture Models** - Mixture distributions (e.g., Gaussian Mixture Models) combine multiple component distributions to model complex, multimodal data. Each component is a simple distribution (e.g., Gaussian), and the overall distribution is a weighted sum.

6. **Latent Variable Models** - In probabilistic PCA, VAEs, and topic models, latent variables follow simple distributions (usually standard Normal or Dirichlet), and the observed data are generated from the latent variables through a conditional distribution.

## Mathematical Explanation

### Discrete Probability Distributions

**Properties of PMF p_X(x)**:
- p_X(x) >= 0 for all x
- sum_x p_X(x) = 1
- P(X in A) = sum_{x in A} p_X(x)

**Bernoulli Distribution**: Models a single binary outcome (success/failure).
- Parameter: p in [0,1] (probability of success)
- PMF: P(X=1) = p, P(X=0) = 1-p
- Mean: E[X] = p, Variance: Var(X) = p(1-p)
- Example: Whether a single email is spam

**Binomial Distribution**: Models the number of successes in n independent Bernoulli trials.
- Parameters: n (trials), p (success probability)
- PMF: P(X=k) = C(n,k) p^k (1-p)^{n-k}, for k = 0, 1, ..., n
- Mean: E[X] = np, Variance: Var(X) = np(1-p)
- Example: Number of heads in 10 coin flips

**Poisson Distribution**: Models the number of events occurring in a fixed interval of time or space.
- Parameter: lambda > 0 (rate parameter, also the mean)
- PMF: P(X=k) = lambda^k e^{-lambda} / k!, for k = 0, 1, 2, ...
- Mean: E[X] = lambda, Variance: Var(X) = lambda
- Example: Number of website visits per minute

### Continuous Probability Distributions

**Properties of PDF f_X(x)**:
- f_X(x) >= 0 for all x
- integral_{-inf}^{inf} f_X(x) dx = 1
- P(a <= X <= b) = integral_a^b f_X(x) dx

**Uniform Distribution**: All values in an interval are equally likely.
- Parameters: a (lower bound), b (upper bound), a < b
- PDF: f(x) = 1/(b-a) for a <= x <= b, 0 otherwise
- Mean: E[X] = (a+b)/2, Variance: Var(X) = (b-a)^2/12
- Example: Random number generation on [0,1]

**Normal (Gaussian) Distribution**: The most important continuous distribution. Bell-shaped and symmetric.
- Parameters: mu (mean), sigma^2 (variance)
- PDF: f(x) = 1/(sigma * sqrt(2pi)) * exp(-(x-mu)^2 / (2 sigma^2))
- Mean: E[X] = mu, Variance: Var(X) = sigma^2
- Standard Normal: Z ~ N(0,1) with CDF denoted Phi(z)
- Example: Measurement errors, heights, test scores

**Exponential Distribution**: Models waiting times between independent events.
- Parameter: lambda > 0 (rate parameter)
- PDF: f(x) = lambda e^{-lambda x} for x >= 0, 0 otherwise
- CDF: F(x) = 1 - e^{-lambda x} for x >= 0
- Mean: E[X] = 1/lambda, Variance: Var(X) = 1/lambda^2
- Memoryless property: P(X > s+t | X > s) = P(X > t)
- Example: Time until next earthquake, battery lifetime

## Formula(s)

1. **Bernoulli PMF**:
   P(X=x) = p^x (1-p)^{1-x}, x in {0,1}

2. **Binomial PMF**:
   P(X=k) = C(n,k) p^k (1-p)^{n-k}

3. **Poisson PMF**:
   P(X=k) = lambda^k e^{-lambda} / k!

4. **Uniform PDF**:
   f(x) = 1/(b-a), a <= x <= b

5. **Normal PDF**:
   f(x) = 1/(sigma sqrt(2pi)) exp(-(x-mu)^2 / (2 sigma^2))

6. **Exponential PDF**:
   f(x) = lambda e^{-lambda x}, x >= 0

7. **CDF Definition**:
   F_X(x) = P(X <= x)

## Properties

1. **Normalisation**: PMF sums to 1; PDF integrates to 1.
2. **CDF range**: 0 <= F_X(x) <= 1.
3. **CDF monotonicity**: F_X is non-decreasing.
4. **CDF limits**: F(x) -> 0 as x -> -inf; F(x) -> 1 as x -> +inf.
5. **Relation between PDF and CDF**: f(x) = d/dx F(x) (for continuous).
6. **Memoryless property**: Exponential is the only continuous distribution with this property.
7. **Stability**: Sum of independent Normals is Normal.
8. **Central Limit Theorem**: Sum of i.i.d. random variables approaches Normal as n increases.
9. **Exponential family**: Bernoulli, Binomial, Poisson, Normal, Exponential are all in the exponential family, enabling generalised linear models.

## Step-by-Step Worked Examples

### Example 1: Binomial Distribution

**Problem**: A fair coin is flipped 8 times. What is the probability of exactly 5 heads? What is the expected number of heads?

**Solution**:

Step 1: Identify the distribution. X ~ Binomial(n=8, p=0.5).

Step 2: Apply the Binomial PMF:
P(X=5) = C(8,5) * (0.5)^5 * (0.5)^3

Step 3: Compute C(8,5) = 8!/(5!3!) = (8*7*6)/(3*2*1) = 56

Step 4: P(X=5) = 56 * (0.5)^8 = 56/256 = 7/32 = 0.21875

Step 5: Expected value: E[X] = np = 8 * 0.5 = 4.

Step 6: Variance: Var(X) = np(1-p) = 8 * 0.5 * 0.5 = 2.

### Example 2: Poisson Distribution

**Problem**: A call center receives an average of 3 calls per minute. What is the probability of exactly 5 calls in a given minute? What is the probability of at most 2 calls?

**Solution**:

Step 1: Identify the distribution. X ~ Poisson(lambda = 3).

Step 2: Apply the Poisson PMF for exactly 5:
P(X=5) = 3^5 * e^{-3} / 5! = 243 * e^{-3} / 120

Step 3: Compute: e^{-3} = 0.04979. So P(X=5) = 243 * 0.04979 / 120 = 12.099 / 120 = 0.1008.

Step 4: At most 2 calls: P(X <= 2) = P(X=0) + P(X=1) + P(X=2).

Step 5: Compute each:
P(X=0) = 3^0 * e^{-3} / 0! = e^{-3} = 0.04979
P(X=1) = 3^1 * e^{-3} / 1! = 3 * 0.04979 = 0.14937
P(X=2) = 3^2 * e^{-3} / 2! = 9 * 0.04979 / 2 = 0.22405

Step 6: P(X <= 2) = 0.04979 + 0.14937 + 0.22405 = 0.4232.

Step 7: Interpret. About 42% of minutes have 0, 1, or 2 calls.

### Example 3: Normal Distribution

**Problem**: IQ scores are normally distributed with mean mu = 100 and standard deviation sigma = 15. What is the probability that a randomly selected person has an IQ between 85 and 115?

**Solution**:

Step 1: Identify the distribution. X ~ N(100, 15^2).

Step 2: Standardise to Z ~ N(0,1). Z = (X - mu) / sigma.

Step 3: For X = 85: Z = (85 - 100) / 15 = -15/15 = -1.
For X = 115: Z = (115 - 100) / 15 = 15/15 = 1.

Step 4: P(85 <= X <= 115) = P(-1 <= Z <= 1).

Step 5: Use standard normal CDF: Phi(1) - Phi(-1).

Step 6: Phi(1) = 0.8413, Phi(-1) = 0.1587.

Step 7: P = 0.8413 - 0.1587 = 0.6826.

Step 8: Interpret. About 68% of people have IQ between 85 and 115 (the empirical rule).

### Example 4: Exponential Distribution

**Problem**: The lifetime of a server follows an Exponential distribution with mean 3 years. What is the probability the server lasts more than 5 years? Given it has lasted 2 years, what is the probability it lasts another 3 years?

**Solution**:

Step 1: Mean = 1/lambda = 3, so lambda = 1/3.

Step 2: For P(X > 5):
P(X > 5) = 1 - F(5) = 1 - (1 - e^{-lambda * 5}) = e^{-5/3} = e^{-1.667} = 0.1889.

Step 3: Memoryless property: P(X > 2+3 | X > 2) = P(X > 3) = e^{-lambda * 3} = e^{-1} = 0.3679.

Step 4: Interpret. There is an 18.9% chance the server lasts more than 5 years. Given it has already lasted 2 years, the probability it lasts another 3 years is 36.8% (same as a new server lasting 3 years).

### Example 5: Bernoulli and Softmax Connection

**Problem**: A logistic regression model outputs logit = 1.2 for a binary classification problem. Compute the predicted probability of the positive class.

**Solution**:

Step 1: The output distribution is Bernoulli with parameter p = sigmoid(logit).

Step 2: Sigmoid function: p = 1 / (1 + e^{-z}) where z = 1.2.

Step 3: p = 1 / (1 + e^{-1.2}) = 1 / (1 + 0.3012) = 1 / 1.3012 = 0.7685.

Step 4: The predicted distribution is Bernoulli(0.7685): P(Y=1) = 0.7685, P(Y=0) = 0.2315.

Step 5: For multi-class with K > 2, the softmax function generalises the sigmoid. For logits z_1, ..., z_K:
p_i = exp(z_i) / sum_j exp(z_j)

Step 6: This produces a categorical distribution over classes, which is the multi-class generalisation of Bernoulli.

## Visual Interpretation

The PMF of a discrete distribution is a bar chart where each bar's height is the probability of that value. The sum of all bar heights is 1. For Binomial with n=10, p=0.5, the bars form a symmetric bell shape centred at 5.

The PDF of a continuous distribution is a smooth curve. The Normal PDF is the classic bell curve. The area under the curve over any interval gives the probability for that interval. The total area is 1.

The CDF is an S-shaped (sigmoid) curve for the Normal distribution, starting at 0 and approaching 1. For the Exponential distribution, the CDF rises quickly and then flattens.

The empirical rule for Normal distributions: approximately 68% of data falls within 1 standard deviation of the mean, 95% within 2, and 99.7% within 3.

## Common Mistakes

1. **Confusing PMF and PDF**: The PMF gives P(X=x) directly for discrete variables. The PDF value f(x) is NOT a probability for continuous variables; it is a density. Only the integral over an interval gives probability.

2. **Applying the wrong distribution**: Using Binomial when trials are not independent, or Poisson when events do not occur at a constant rate. Always check the assumptions.

3. **Misinterpreting the Normal approximation**: The normal approximation to the Binomial works well only when np >= 10 and n(1-p) >= 10. Applying it with small n or extreme p leads to poor results.

4. **Ignoring the memoryless property of Exponential**: The Exponential is the only continuous distribution where P(X > s+t | X > s) = P(X > t). This means the remaining waiting time does not depend on how long you have already waited.

5. **Forgetting that the CDF is P(X <= x), not P(X < x)**: For discrete distributions, this distinction matters. P(X <= x) includes the value x; P(X < x) does not.

6. **Assuming independence when using Binomial**: The Binomial requires independent trials. Drawing without replacement from a finite population requires the Hypergeometric distribution instead.

7. **Confusing the parameterisation of Exponential**: Some texts parameterise Exponential as f(x) = (1/beta) e^{-x/beta} where beta = 1/lambda is the mean. Always check which parameterisation is being used.

## Interview Questions

### Beginner

1. **Q**: What is the difference between a PMF and a PDF?
   **A**: A PMF gives P(X=x) for discrete random variables. A PDF gives probability density for continuous variables; probability is the integral of the PDF over an interval.

2. **Q**: What are the parameters of the Binomial distribution?
   **A**: n (number of trials) and p (probability of success on each trial).

3. **Q**: State the empirical rule for Normal distributions.
   **A**: About 68% of data falls within 1 sigma of the mean, 95% within 2 sigma, and 99.7% within 3 sigma.

4. **Q**: What distribution models the number of events in a fixed interval of time when events occur independently at a constant average rate?
   **A**: The Poisson distribution.

5. **Q**: What is the mean and variance of a Bernoulli(p) random variable?
   **A**: Mean = p, Variance = p(1-p).

### Intermediate

1. **Q**: Explain the relationship between the Poisson and Binomial distributions.
   **A**: The Poisson(lambda) distribution is the limit of Binomial(n, p) as n -> inf and p -> 0 with np = lambda. It approximates the Binomial for large n and small p.

2. **Q**: In machine learning, what distribution does the softmax function produce?
   **A**: The softmax function produces a categorical distribution. The output is a vector of probabilities summing to 1, where each entry is P(Y = class i).

3. **Q**: What is the memoryless property of the Exponential distribution?
   **A**: P(X > s+t | X > s) = P(X > t). The remaining waiting time is independent of how long you have already waited.

4. **Q**: Why is the Normal distribution so important in statistics and ML?
   **A**: Due to the Central Limit Theorem: sums of i.i.d. random variables approach normality regardless of the original distribution. This justifies using Normal approximations and explains why many natural phenomena are approximately normal.

5. **Q**: How does the choice of output distribution determine the loss function in a neural network?
   **A**: By assuming an output distribution (e.g., Gaussian for regression, Bernoulli for binary classification, Categorical for multi-class), the loss function is the negative log-likelihood: MSE for Gaussian, binary cross-entropy for Bernoulli, categorical cross-entropy for Categorical.

### Advanced

1. **Q**: Derive the MLE for the parameter p of a Bernoulli distribution.
   **A**: For data x_1, ..., x_n, the likelihood is L(p) = product_i p^{x_i} (1-p)^{1-x_i}. The log-likelihood is l(p) = (sum x_i) log p + (n - sum x_i) log(1-p). Setting derivative to 0 gives p_hat = (sum x_i)/n, the sample proportion.

2. **Q**: Explain how Gaussian processes define a distribution over functions.
   **A**: A GP is a collection of random variables where any finite subset has a multivariate normal distribution. It is specified by a mean function m(x) and covariance function k(x,x'). For any finite set of input points, f = [f(x_1), ..., f(x_n)] ~ N(m, K) where K_ij = k(x_i, x_j). This defines a prior over functions, and conditioning on observed data yields a posterior over functions.

3. **Q**: Prove that the sum of two independent Poisson random variables is Poisson.
   **A**: Let X ~ Poisson(lambda_1) and Y ~ Poisson(lambda_2). Then P(X+Y = k) = sum_{i=0}^k P(X=i) P(Y=k-i) = sum_{i=0}^k [lambda_1^i e^{-lambda_1}/i!] * [lambda_2^{k-i} e^{-lambda_2}/(k-i)!] = e^{-(lambda_1+lambda_2)}/k! * sum_{i=0}^k k!/(i!(k-i)!) lambda_1^i lambda_2^{k-i} = e^{-(lambda_1+lambda_2)} (lambda_1+lambda_2)^k / k!, which is Poisson(lambda_1+lambda_2).

## Practice Problems

### Easy

1. Let X ~ Binomial(n=5, p=0.3). Find P(X=2).

2. Let X ~ Poisson(lambda=2). Find P(X=3).

3. Let Z ~ N(0,1). Find P(Z < 1.5).

4. Let X ~ Uniform(0, 10). Find P(X > 6).

5. Let X ~ Exponential(lambda=0.5). Find P(X > 2).

### Medium

1. A multiple-choice test has 10 questions, each with 4 options. A student guesses randomly. What is the probability they get exactly 7 correct?

2. A factory produces light bulbs with a 2% defect rate. In a batch of 200, what is the probability of exactly 5 defects? (Use Poisson approximation.)

3. IQ scores are N(100, 15^2). What IQ score corresponds to the 90th percentile?

4. The time between customer arrivals is Exponential with mean 5 minutes. What is the probability the next customer arrives within 3 minutes?

5. A random variable X has CDF F(x) = 1 - e^{-2x} for x >= 0. Find the PDF and compute P(0.5 < X < 1).

### Hard

1. Derive the mean and variance of the Binomial distribution using the fact that it is the sum of independent Bernoulli random variables.

2. Let X ~ N(0,1). Show that Y = X^2 follows a chi-square distribution with 1 degree of freedom. What is its PDF?

3. Prove that the Exponential distribution is the unique continuous distribution with the memoryless property.

## Solutions

### Easy Solutions

**Solution 1**: P(X=2) = C(5,2) * 0.3^2 * 0.7^3 = 10 * 0.09 * 0.343 = 0.3087.

**Solution 2**: P(X=3) = 2^3 * e^{-2} / 6 = 8 * 0.1353 / 6 = 0.1804.

**Solution 3**: P(Z < 1.5) = Phi(1.5) = 0.9332.

**Solution 4**: P(X > 6) = (10-6)/10 = 4/10 = 0.4.

**Solution 5**: P(X > 2) = e^{-0.5*2} = e^{-1} = 0.3679.

### Medium Solutions

**Solution 1**: X ~ Binomial(10, 0.25). P(X=7) = C(10,7) * 0.25^7 * 0.75^3 = 120 * 0.000061 * 0.422 = 0.00309.

**Solution 2**: lambda = np = 200 * 0.02 = 4. P(X=5) = 4^5 * e^{-4} / 120 = 1024 * 0.0183 / 120 = 0.1563.

**Solution 3**: Need z such that Phi(z) = 0.9. From tables, z = 1.28. Score = mu + z*sigma = 100 + 1.28*15 = 119.2.

**Solution 4**: lambda = 1/5 = 0.2. P(X < 3) = 1 - e^{-0.2*3} = 1 - e^{-0.6} = 1 - 0.5488 = 0.4512.

**Solution 5**: f(x) = d/dx F(x) = 2e^{-2x} for x >= 0. P(0.5 < X < 1) = F(1) - F(0.5) = (1-e^{-2}) - (1-e^{-1}) = e^{-1} - e^{-2} = 0.3679 - 0.1353 = 0.2326.

### Hard Solutions

**Solution 1**: X = sum_{i=1}^n Y_i where Y_i ~ Bernoulli(p). E[X] = sum E[Y_i] = n * p. Var(X) = sum Var(Y_i) = n * p(1-p) (by independence).

**Solution 2**: For Y = X^2, the CDF F_Y(y) = P(X^2 <= y) = P(-sqrt(y) <= X <= sqrt(y)) = Phi(sqrt(y)) - Phi(-sqrt(y)) = 2Phi(sqrt(y)) - 1. PDF: f_Y(y) = d/dy [2Phi(sqrt(y))] = 2 * phi(sqrt(y)) * 1/(2sqrt(y)) = 1/(sqrt(2pi y)) e^{-y/2}. This is chi-square(1).

**Solution 3**: Memoryless: P(X > s+t | X > s) = P(X > t). Equivalently, P(X > s+t) = P(X > s)P(X > t). Let S(t) = P(X > t). Then S(s+t) = S(s)S(t). For continuous S, the only solution is S(t) = e^{-lambda t} for some lambda > 0. Therefore P(X > t) = e^{-lambda t}, which implies the PDF f(t) = lambda e^{-lambda t}, the Exponential distribution.

## Related Concepts

- **Random Variable (MATH-070)**: The object whose distribution is being described
- **Probability (MATH-065)**: The foundation of distribution theory
- **Conditional Probability (MATH-068)**: Conditional distributions
- **Bayes Theorem (MATH-069)**: Posterior distributions
- **Expected Value**: The mean of a distribution
- **Variance**: A measure of spread of a distribution
- **Central Limit Theorem**: Why the Normal distribution is ubiquitous

## Next Concepts

- **Joint Distributions**: Distributions of multiple random variables
- **Marginal and Conditional Distributions**: Decomposing joint distributions
- **Law of Large Numbers**: Convergence of sample means
- **Central Limit Theorem**: The foundation of statistical inference
- **Exponential Family**: A unified framework for distributions

## Summary

A probability distribution completely describes the probabilities of a random variable's possible values. Discrete distributions (Bernoulli, Binomial, Poisson) are described by PMFs; continuous distributions (Uniform, Normal, Exponential) are described by PDFs. The CDF F(x) = P(X <= x) unifies both types. Each distribution has specific properties, parameters, and real-world applications. In machine learning, distributions appear as model outputs (softmax = categorical), loss functions (negative log-likelihood), and generative models. Understanding the key distributions and their properties is essential for modelling, inference, and decision-making under uncertainty.

## Key Takeaways

- A probability distribution fully characterises a random variable
- Discrete: PMF p(x) = P(X=x); Continuous: PDF f(x) integrates to probability
- CDF F(x) = P(X <= x) works for both discrete and continuous
- Bernoulli models a single binary outcome; Binomial models count of successes
- Poisson models rare events in fixed intervals
- Normal is the most important distribution (bell curve, CLT)
- Uniform models equal-probability continuous outcomes
- Exponential models waiting times with memoryless property
- Softmax outputs a categorical distribution over classes
- Loss functions correspond to negative log-likelihood under assumed output distributions
- The sum of independent Normals is Normal; sum of independent Poissons is Poisson
- Always verify distribution assumptions before applying a model
