# Concept: Bayes Theorem

## Concept ID

MATH-069

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- State Bayes theorem and identify its four components: prior, likelihood, evidence, posterior
- Derive Bayes theorem from the definition of conditional probability
- Apply Bayes theorem to compute posterior probabilities from data
- Implement a Naive Bayes classifier from first principles
- Understand how Bayesian inference differs from frequentist inference
- Recognise the role of Bayes theorem in modern machine learning

## Prerequisites

- Conditional probability (MATH-068)
- Law of total probability
- Basic probability axioms (MATH-065)

## Definition

**Bayes theorem** describes the probability of an event based on prior knowledge of conditions that might be related to the event. It provides a mathematical rule for inverting conditional probabilities:

P(A|B) = P(B|A) P(A) / P(B)

where:
- P(A) is the **prior probability** of A (our belief before seeing evidence)
- P(B|A) is the **likelihood** of B given A (the probability of the evidence assuming A is true)
- P(B) is the **evidence** or marginal probability of B
- P(A|B) is the **posterior probability** of A given B (our updated belief after seeing evidence)

## Intuition

Bayes theorem formalises how we learn from experience. Before seeing evidence, we have a prior belief about a hypothesis. When we observe evidence, we update that belief into a posterior belief. The update is proportional to how well the hypothesis explains the evidence (likelihood) weighted by how plausible the hypothesis was to begin with (prior).

If the evidence is very likely under a hypothesis, that hypothesis gains credibility. If the evidence is unlikely under a hypothesis, that hypothesis loses credibility. The evidence term P(B) normalises the result, ensuring the posterior probabilities over all hypotheses sum to 1.

## Why This Concept Matters

Bayes theorem is arguably the single most important formula in data science and machine learning. It provides a principled framework for learning from data: start with prior beliefs, observe data, update to posterior beliefs. This framework underpins Bayesian inference, probabilistic graphical models, Bayesian neural networks, and many state-of-the-art AI systems. Understanding Bayes theorem is essential for interpreting medical test results, building spam filters, performing A/B testing, and quantifying uncertainty in predictions.

## Historical Background

Bayes theorem is named after **Thomas Bayes** (1702-1761), a Presbyterian minister and mathematician. His work *An Essay Towards Solving a Problem in the Doctrine of Chances* was published posthumously in 1763 by his friend Richard Price. The essay contained the first statement of what we now call Bayes theorem.

The theorem was independently discovered and popularised by **Pierre-Simon Laplace** (1749-1827), who used it extensively in celestial mechanics, demography, and jurisprudence. Laplace proved the first general version of the theorem and demonstrated its practical applications.

Bayesian statistics declined in the early 20th century due to the rise of frequentist statistics championed by **Ronald Fisher** (1890-1962) and **Jerzy Neyman** (1894-1981). However, it experienced a renaissance with the development of Markov Chain Monte Carlo (MCMC) methods that made Bayesian computation tractable. Today, Bayesian methods are central to machine learning, artificial intelligence, and many scientific fields.

## Real World Examples

1. **Medical Testing** - A patient tests positive for a rare disease. Bayes theorem combines the test accuracy (likelihood) with the disease prevalence (prior) to compute the probability that the patient actually has the disease (posterior). This is essential for interpreting test results correctly and avoiding the base rate fallacy.

2. **Spam Filtering** - Email spam filters use Bayes theorem to compute P(spam|email words). The prior is the base spam rate, the likelihood is how frequently each word appears in spam versus legitimate email. This is the Naive Bayes classifier in action.

3. **Legal Reasoning** - In court, Bayes theorem connects the probability of guilt given evidence P(guilty|evidence) to the probability of finding the evidence if the defendant were guilty P(evidence|guilty) and the prior probability of guilt.

4. **Search Engines** - When you type a query, the search engine computes P(document|query) using a Bayesian framework, ranking documents by their posterior relevance given the query terms.

5. **A/B Testing** - Companies running A/B tests use Bayesian methods to compute P(B is better than A | observed data), allowing them to make probabilistic decisions about which version to deploy.

## AI/ML Relevance

1. **Naive Bayes Classifier** - The most direct application of Bayes theorem in ML. Given features x_1, ..., x_d and class y, the classifier computes:
   P(y|x_1, ..., x_d) proportional to P(y) * product_j P(x_j|y)
   The naive assumption is conditional independence of features given the class. Despite this strong assumption, Naive Bayes works well for text classification, spam filtering, and sentiment analysis.

2. **Bayesian Neural Networks** - Instead of learning point estimates of weights, BNNs learn a posterior distribution P(theta|D) over weights. Predictions are made by integrating over possible weights, naturally providing uncertainty estimates.

3. **Bayesian Optimisation** - A sequential strategy for optimising expensive black-box functions. It uses a Gaussian process prior over functions and Bayes theorem to update the posterior after each evaluation. The acquisition function guides the next evaluation point.

4. **Probabilistic Graphical Models** - Bayesian networks use Bayes theorem for inference: given observed variables, compute the posterior distribution of unobserved variables.

5. **Reinforcement Learning** - Bayesian RL maintains a posterior distribution over environment parameters or value functions. The agent explores to reduce uncertainty about unknown parameters through Thompson sampling.

6. **Variational Autoencoders** - VAEs learn a posterior distribution P(z|x) over latent variables z given data x. Since the true posterior is intractable, they approximate it with a variational distribution q(z|x) by maximising the Evidence Lower Bound (ELBO).

## Mathematical Explanation

### Derivation of Bayes Theorem

Starting from the definition of conditional probability:
P(A|B) = P(A cap B) / P(B) and P(B|A) = P(A cap B) / P(A)

From the second equation, P(A cap B) = P(B|A) P(A).

Substitute into the first equation:
P(A|B) = P(B|A) P(A) / P(B)

To compute P(B), use the law of total probability. If the hypotheses A_1, A_2, ..., A_n partition the sample space:
P(B) = sum_i P(B|A_i) P(A_i)

This gives the full form:
P(A_i|B) = P(B|A_i) P(A_i) / sum_j P(B|A_j) P(A_j)

### Components in Detail

**Prior P(A)**: The initial probability of hypothesis A before seeing data. Priors can be objective (e.g., uniform), weakly informative, or informative (based on previous studies).

**Likelihood P(B|A)**: The probability of observing the evidence B assuming hypothesis A is true.

**Evidence P(B)**: The total probability of the evidence across all hypotheses. It normalises the posterior.

**Posterior P(A|B)**: The updated probability of hypothesis A after seeing evidence B. This becomes the prior for future updates.

### Bayesian Updating as Learning

Bayes theorem supports sequential learning:
P(theta | D_1, D_2) proportional to P(D_2 | theta, D_1) P(theta | D_1)

The posterior from the first dataset becomes the prior for the second dataset. This sequential nature matches how humans learn.

## Formula(s)

1. **Bayes Theorem (Simple Form)**:
   P(A|B) = P(B|A) P(A) / P(B)

2. **Bayes Theorem (General Form with Partition)**:
   P(A_i|B) = P(B|A_i) P(A_i) / sum_j P(B|A_j) P(A_j)

3. **Posterior Proportionality**:
   P(A|B) proportional to P(B|A) P(A)

4. **Sequential Bayesian Updating**:
   P(theta|D_1, D_2) proportional to P(D_2|theta, D_1) P(theta|D_1)

5. **Bayesian Prediction (Posterior Predictive)**:
   P(x_tilde | D) = integral P(x_tilde | theta) P(theta | D) d theta

## Properties

1. **Proportionality**: The posterior is proportional to the likelihood times the prior.
2. **Normalisation**: The denominator ensures sum_i P(A_i|B) = 1.
3. **Commutativity of Evidence**: The order of evidence does not matter.
4. **Consistency**: As the sample size grows, the posterior concentrates around the true parameter value.
5. **Conjugacy**: For certain prior-likelihood pairs, the posterior has the same distribution family as the prior.
6. **Asymptotic Normality**: Under regularity conditions, the posterior approaches a Gaussian for large samples.

## Step-by-Step Worked Examples

### Example 1: Medical Test (Base Rate Fallacy)

**Problem**: A disease affects 1% of the population. A test is 95% sensitive (positive if diseased) and 90% specific (negative if not diseased). If a person tests positive, what is the probability they actually have the disease?

**Solution**:

Step 1: Define events. D = has disease, T = tests positive.

Step 2: Write down the given probabilities:
- Prior: P(D) = 0.01, P(D^c) = 0.99
- Likelihood: P(T|D) = 0.95, P(T|D^c) = 0.10

Step 3: Compute the evidence P(T) using the law of total probability:
P(T) = P(T|D) P(D) + P(T|D^c) P(D^c) = 0.95 * 0.01 + 0.10 * 0.99 = 0.0095 + 0.099 = 0.1085

Step 4: Apply Bayes theorem:
P(D|T) = P(T|D) P(D) / P(T) = 0.0095 / 0.1085 = 0.0876

Step 5: Interpret. Even with a positive test, there is only an 8.76% chance of having the disease. This is due to the low prevalence.

### Example 2: Naive Bayes Text Classification

**Problem**: Classify an email as spam (S) or not spam (N). Training data: 40% of emails are spam. The word "free" appears in 30% of spam and 5% of non-spam. The word "money" appears in 20% of spam and 2% of non-spam. An email contains both "free" and "money". What is P(S|free, money)?

**Solution**:

Step 1: Define events. S = spam, N = not spam.

Step 2: Priors: P(S) = 0.4, P(N) = 0.6.

Step 3: Likelihoods:
- P(free|S) = 0.30, P(free|N) = 0.05
- P(money|S) = 0.20, P(money|N) = 0.02

Step 4: Naive Bayes assumption: words are conditionally independent given the class.
P(free, money|S) = P(free|S) * P(money|S) = 0.30 * 0.20 = 0.06
P(free, money|N) = 0.05 * 0.02 = 0.001

Step 5: Compute the unnormalised posterior:
P(S|free, money) proportional to 0.06 * 0.4 = 0.024
P(N|free, money) proportional to 0.001 * 0.6 = 0.0006

Step 6: Normalise:
P(S|free, money) = 0.024 / (0.024 + 0.0006) = 0.024 / 0.0246 = 0.9756

Step 7: Interpret. There is approximately a 97.6% chance the email is spam.

### Example 3: Bayesian Parameter Estimation (Beta-Bernoulli)

**Problem**: A coin is flipped 10 times and shows heads 7 times. Using a Beta(2,2) prior, compute the posterior distribution for the probability of heads theta. What is the posterior mean?

**Solution**:

Step 1: Model. X_i ~ Bernoulli(theta), i = 1, ..., 10. Number of heads y = 7.

Step 2: Likelihood:
P(y|theta) = C(10,7) theta^7 (1-theta)^3 proportional to theta^7 (1-theta)^3

Step 3: Prior: theta ~ Beta(2, 2), so p(theta) proportional to theta^{1} (1-theta)^{1}.

Step 4: Posterior:
p(theta|y) proportional to theta^7 (1-theta)^3 * theta (1-theta) = theta^8 (1-theta)^4

Step 5: This is Beta(9, 5).

Step 6: Posterior mean:
E[theta|y] = 9 / (9+5) = 9/14 = 0.643

Step 7: Interpret. The posterior mean is 0.643, between the prior mean of 0.5 and the sample proportion of 0.7.

### Example 4: Multiple Hypotheses with Bayes

**Problem**: A bag contains one of three types: Type A (3 red, 2 blue), Type B (4 red, 1 blue), Type C (1 red, 4 blue). Prior probabilities are equal. One marble is drawn and is red. What is the posterior probability for each type?

**Solution**:

Step 1: Priors: P(A) = P(B) = P(C) = 1/3.

Step 2: Likelihoods:
- P(red|A) = 3/5 = 0.6
- P(red|B) = 4/5 = 0.8
- P(red|C) = 1/5 = 0.2

Step 3: Evidence P(red) = (0.6 + 0.8 + 0.2)/3 = 1.6/3

Step 4: Posteriors:
P(A|red) = 0.6/1.6 = 0.375
P(B|red) = 0.8/1.6 = 0.5
P(C|red) = 0.2/1.6 = 0.125

Step 5: After seeing a red marble, Type B is the most probable (50%).

### Example 5: Sequential Bayesian Updating

**Problem**: Starting from a Beta(2,2) prior, a coin shows heads then tails. Show that sequential updating gives the same result as batch updating.

**Solution**:

Step 1: Prior Beta(2,2): p(theta) proportional to theta^1 (1-theta)^1.

Step 2: After heads: likelihood proportional to theta. Posterior = Beta(3, 2).

Step 3: After tails: likelihood proportional to (1-theta). Posterior = Beta(3, 3).

Step 4: Batch update with (heads, tails): likelihood = theta (1-theta). Posterior = Beta(3, 3).

Step 5: Results match. Sequential and batch updating are equivalent.

## Visual Interpretation

Bayes theorem can be visualised as a probability tree. The first branches represent prior probabilities of hypotheses, second-level branches represent likelihood of evidence under each hypothesis. The posterior for a specific hypothesis is the probability of that hypothesis's path divided by the total evidence probability.

A second visualisation is the "Bayesian updating" diagram: a distribution shifting from prior (broad, uncertain) to posterior (narrower, more certain) as evidence accumulates.

## Common Mistakes

1. **Confusing P(A|B) with P(B|A)** (Prosecutor's Fallacy): P(disease|positive) is not the same as P(positive|disease).

2. **Ignoring the prior** (Base Rate Fallacy): A test can be 99% accurate, but if the disease affects 0.1% of the population, P(disease|positive) is still low.

3. **Forgetting to normalise**: Computing the numerator P(B|A)P(A) but forgetting to divide by P(B).

4. **Assuming uniform priors without justification**: A uniform prior is not always appropriate.

5. **Misinterpreting the posterior as the only truth**: The posterior depends on the prior. Different reasonable priors can lead to different posteriors with small data.

6. **Applying Bayes theorem to non-disjoint hypotheses**: The general form requires the hypotheses to partition the sample space.

7. **Believing Bayesian and frequentist methods always agree**: They often agree with large data but can differ with small data.

## Interview Questions

### Beginner

1. **Q**: State Bayes theorem.
   **A**: P(A|B) = P(B|A)P(A) / P(B).

2. **Q**: What are the four components of Bayes theorem?
   **A**: Prior P(A), likelihood P(B|A), evidence P(B), and posterior P(A|B).

3. **Q**: In medical testing, why might a positive test not guarantee disease?
   **A**: Due to the base rate fallacy. If the disease is rare (low prior), even a highly accurate test produces many false positives.

4. **Q**: What is the difference between a prior and a posterior?
   **A**: The prior is the probability of a hypothesis before seeing data. The posterior is the updated probability after seeing data.

5. **Q**: If P(A) = 0.3, P(B|A) = 0.8, and P(B) = 0.5, compute P(A|B).
   **A**: P(A|B) = (0.8 * 0.3) / 0.5 = 0.24 / 0.5 = 0.48.

### Intermediate

1. **Q**: Derive Bayes theorem from conditional probability.
   **A**: From P(A|B) = P(A cap B)/P(B) and P(B|A) = P(A cap B)/P(A), we get P(A|B) = P(B|A)P(A)/P(B).

2. **Q**: What is the Naive Bayes assumption?
   **A**: Features are conditionally independent given the class. This rarely holds in practice yet the classifier often performs well.

3. **Q**: Explain how Bayes theorem is used in Bayesian optimisation.
   **A**: Uses a Gaussian process prior over the objective function. After each evaluation, Bayes theorem updates the posterior, and an acquisition function uses the posterior to decide where to evaluate next.

4. **Q**: What is a conjugate prior and why is it useful?
   **A**: A prior that, combined with a likelihood, yields a posterior in the same distribution family. Simplifies computation.

5. **Q**: In Naive Bayes, what happens when a word appears in the test set but not in training for a given class?
   **A**: The conditional probability would be zero, making the entire posterior zero. Laplace smoothing adds a small pseudo-count to avoid this.

### Advanced

1. **Q**: Prove that the posterior of a Gaussian likelihood with Gaussian prior is Gaussian, deriving the posterior mean and variance.
   **A**: For y_i|theta ~ N(theta, sigma^2) and theta ~ N(mu_0, tau_0^2), the posterior is theta|y ~ N(mu_n, tau_n^2) where mu_n = (mu_0/tau_0^2 + n*y_bar/sigma^2) / (1/tau_0^2 + n/sigma^2) and tau_n^2 = 1/(1/tau_0^2 + n/sigma^2).

2. **Q**: Explain the Evidence Lower Bound (ELBO) and its connection to Bayes theorem.
   **A**: log P(x) = KL(q(z) || P(z|x)) + ELBO(q). Maximising ELBO minimises the KL divergence between the variational approximation q(z) and the true posterior P(z|x), avoiding computation of the intractable P(x).

3. **Q**: Discuss limitations of Bayes theorem in high dimensions and how MCMC addresses them.
   **A**: Computing the evidence P(data) = integral P(data|theta) P(theta) dtheta is intractable in high dimensions. MCMC methods avoid computing P(data) by sampling from the unnormalised posterior, constructing a Markov chain whose stationary distribution is the posterior.

## Practice Problems

### Easy

1. P(A) = 0.2, P(B|A) = 0.9, P(B|A^c) = 0.3. Find P(A|B).

2. In a population, 10% have a disease. A test is 90% sensitive and 85% specific. Find P(disease|positive).

3. A spam filter knows 20% of emails are spam. The word "offer" appears in 40% of spam and 5% of non-spam. If an email contains "offer", what is P(spam|offer)?

4. Two coins: one fair, one double-headed. A coin is chosen at random and flipped, showing heads. What is P(fair|heads)?

5. P(A|B) = 0.7, P(B) = 0.4, P(A) = 0.5. Find P(B|A).

### Medium

1. Three machines: M1 (30% output, 2% defective), M2 (50%, 1% defective), M3 (20%, 3% defective). A defective item is found. Find P(M2|defective).

2. A bag has 5 red and 3 blue marbles. A marble is drawn and replaced. A second marble is drawn and is red. Find P(first was red | second is red).

3. Derive the odds form: posterior odds = prior odds * likelihood ratio.

4. In court, P(evidence|guilty) = 0.95, P(evidence|innocent) = 0.10, prior P(guilty) = 0.01. Find P(guilty|evidence).

5. A test has 99% sensitivity and 95% specificity. Disease prevalence is 0.5%. A person tests positive twice (independent tests). Find P(disease|two positives).

### Hard

1. Prove that Bayesian updating is commutative: updating with evidence E1 then E2 gives the same posterior as E2 then E1.

2. Derive the posterior for a Poisson likelihood with Gamma prior, showing the Gamma is conjugate.

3. For Bayesian linear regression y = X beta + epsilon, epsilon ~ N(0, sigma^2 I), prior beta ~ N(mu_0, Sigma_0), derive the posterior p(beta|y,X).

## Solutions

### Easy Solutions

**Solution 1**: P(B) = 0.9*0.2 + 0.3*0.8 = 0.42. P(A|B) = 0.18/0.42 = 3/7.

**Solution 2**: P(D) = 0.1, P(T|D) = 0.9, P(T|D^c) = 0.15. P(D|T) = (0.9*0.1)/(0.9*0.1+0.15*0.9) = 0.09/0.225 = 0.4.

**Solution 3**: P(S) = 0.2, P(O|S) = 0.4, P(O|N) = 0.05. P(S|O) = (0.4*0.2)/(0.4*0.2+0.05*0.8) = 0.08/0.12 = 2/3.

**Solution 4**: P(F|H) = (0.5*0.5)/(0.5*0.5+1*0.5) = 0.25/0.75 = 1/3.

**Solution 5**: P(B|A) = 0.7*0.4/0.5 = 0.56.

### Medium Solutions

**Solution 1**: P(D) = 0.02*0.3+0.01*0.5+0.03*0.2 = 0.017. P(M2|D) = 0.005/0.017 = 5/17.

**Solution 2**: P(R1|R2) = (5/8)*(5/8)/(5/8) = 5/8 = 0.625.

**Solution 3**: P(A|B)/P(A^c|B) = [P(B|A)/P(B|A^c)] * [P(A)/P(A^c)].

**Solution 4**: P(G|E) = (0.95*0.01)/(0.95*0.01+0.10*0.99) = 0.0095/0.1085 = 0.0876.

**Solution 5**: After first test: P(D|T1) = 0.0905. After second: P(D|T1,T2) = (0.99*0.0905)/(0.99*0.0905+0.05*0.9095) = 0.663.

### Hard Solutions

**Solution 1**: P(theta|E1,E2) proportional to P(E2|theta)P(E1|theta)P(theta) which is symmetric in E1 and E2.

**Solution 2**: Poisson: P(y|lambda) = lambda^y e^{-lambda}/y!. Gamma prior: p(lambda) proportional to lambda^{a-1} e^{-b lambda}. Posterior: Gamma(a+y, b+1).

**Solution 3**: Posterior is Gaussian with mean mu_n = Sigma_n (X^T y/sigma^2 + Sigma_0^{-1} mu_0) and covariance Sigma_n = (X^T X/sigma^2 + Sigma_0^{-1})^{-1}.

## Related Concepts

- **Conditional Probability (MATH-068)**: The foundation of Bayes theorem
- **Probability (MATH-065)**: The broader framework
- **Random Variable (MATH-070)**: Bayesian inference over random parameters
- **Probability Distribution (MATH-071)**: Prior and posterior distributions

## Next Concepts

- **Random Variable**: Representing uncertain quantities numerically (MATH-070)
- **Probability Distribution**: Formal descriptions of random variables (MATH-071)
- **Bayesian Inference**: The broader methodology built on Bayes theorem

## Summary

Bayes theorem P(A|B) = P(B|A)P(A)/P(B) provides a mathematical rule for updating beliefs based on evidence. It combines prior knowledge with the likelihood of evidence to produce a posterior belief. Bayes theorem is the foundation of the Naive Bayes classifier, Bayesian neural networks, Bayesian optimisation, and probabilistic graphical models. It enables principled uncertainty quantification, sequential learning, and decision-making under uncertainty.

## Key Takeaways

- Bayes theorem inverts conditional probabilities: P(A|B) = P(B|A)P(A)/P(B)
- The prior encodes beliefs before seeing data
- The likelihood measures how well the hypothesis explains the evidence
- The posterior is the updated belief after observing data
- The evidence normalises the posterior to sum to 1
- The Naive Bayes classifier assumes feature independence given the class
- Bayesian methods provide natural uncertainty quantification
- Conjugate priors simplify Bayesian computation
- Sequential and batch updating are equivalent
- The base rate fallacy illustrates why priors matter
- MCMC enables Bayesian inference in complex models
