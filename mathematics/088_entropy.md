# Concept: Entropy

## Concept ID

MATH-088

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Information Theory

## Learning Objectives

- Define Shannon entropy as a measure of uncertainty in a random variable
- Compute the entropy of discrete probability distributions in both bits and nats
- Understand the relationship between entropy, uniform distributions, and maximum entropy
- Apply entropy in AI/ML contexts such as decision tree splitting and regularisation
- Distinguish between entropy and other information-theoretic quantities (cross-entropy, KL divergence)

## Prerequisites

- Probability (MATH-065)
- Random Variable (MATH-070)
- Probability Distribution (MATH-071)
- Logarithms and exponentials (algebraic manipulation)
- Basic summation notation

## Definition

**Entropy** (also called **Shannon entropy**) quantifies the average amount of information or uncertainty associated with the outcomes of a random variable. For a discrete random variable $X$ with probability mass function $P(x_i) = P(X = x_i)$, the entropy $H(X)$ is defined as:

$$
H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
$$

measured in **bits** (shannons). Using the natural logarithm gives entropy in **nats**:

$$
H(X) = -\sum_{i=1}^{n} P(x_i) \ln P(x_i)
$$

By convention, $0 \log 0 = 0$, since $\lim_{p \to 0^+} p \log p = 0$.

Entropy represents the minimum number of bits required on average to encode outcomes of $X$ under an optimal coding scheme. It is the fundamental limit of lossless compression for a given distribution.

## Intuition

Entropy measures surprise or uncertainty. A deterministic event (e.g., a coin with two heads) has zero entropy because there is no uncertainty — you already know the outcome. A fair coin has maximal entropy among binary distributions because each flip is maximally surprising: you have no information about which side will appear.

Think of entropy as the number of yes/no questions needed on average to determine the outcome. For a fair coin, you need exactly 1 bit per flip. For a biased coin that lands heads 99% of the time, you need fewer than 1 bit on average because you can guess "heads" and be right most of the time, only needing follow-up questions in rare cases.

The logarithm in entropy reflects the intuition that information is additive for independent events. If you flip two independent fair coins, there are 4 equally likely outcomes, requiring $2 = \log_2 4$ bits. Entropy grows logarithmically with the number of equally likely outcomes.

## Why This Concept Matters

Entropy is a foundational concept in information theory, statistics, physics, and machine learning:

- **Lossless compression:** Entropy sets the theoretical lower bound on how much data can be compressed. Huffman coding, arithmetic coding, and Lempel-Ziv algorithms all aim to achieve entropy-rate compression.
- **Coding theory:** Entropy quantifies the information capacity of a channel (Shannon's noisy channel coding theorem).
- **Statistical inference:** Maximum entropy models (MaxEnt) select the least informative distribution consistent with observed constraints, providing a principled approach to modelling.
- **Machine learning:** Entropy appears in decision tree splitting criteria, loss functions, regularisation terms, and evaluation metrics.
- **Physics:** Thermodynamic entropy is proportional to Shannon entropy via Boltzmann's constant, linking information theory to statistical mechanics.
- **Neuroscience:** The brain can be modelled as an information-processing system; entropy quantifies neural coding efficiency.

## Historical Background

**Claude Shannon** (1916–2001) introduced entropy in his landmark 1948 paper *"A Mathematical Theory of Communication"* published in the Bell System Technical Journal. Shannon was solving the practical problem of efficient signal transmission over noisy telephone lines at Bell Labs.

Shannon's key insight was that information could be measured in bits (binary digits), a term he credited to John Tukey. He derived entropy from three natural axioms that any measure of information should satisfy: continuity, monotonicity with number of outcomes, and additivity for independent events. The only function satisfying these axioms is the negative sum of $p_i \log p_i$.

Shannon consulted John von Neumann on what to call his measure. Von Neumann reportedly said: "Call it entropy. It is already in use under that name, and besides, it will give you a great edge in debates because nobody really knows what entropy is anyway." The name stuck, linking Shannon's information measure to Boltzmann's thermodynamic entropy $S = k_B \ln W$.

**Ralph Hartley** (1928) had previously proposed $H = \log N$ for equally likely outcomes, which Shannon generalised. **Andrey Kolmogorov** later developed algorithmic information theory, providing another perspective on entropy through description length.

## Real World Examples

1. **Coin flips:** A fair coin ($P(H) = P(T) = 0.5$) has $H = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1$ bit. A biased coin ($P(H) = 0.9, P(T) = 0.1$) has $H = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.469$ bits — less uncertainty.

2. **Dice rolls:** A fair 6-sided die has $H = \log_2 6 \approx 2.585$ bits per roll. A loaded die with probabilities $[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]$ has $H \approx 2.161$ bits.

3. **Weather prediction:** A city where it rains 50% of days and is sunny 50% of days has $H = 1$ bit. A desert city with 5% rain probability has $H \approx 0.286$ bits — much less uncertainty about daily weather.

4. **Text compression:** English text has an empirical entropy of roughly 1.0–1.5 bits per character (depending on the model), far below the 8 bits per character in ASCII. This gap explains why compression algorithms like gzip can reduce text files to 20-30% of their original size.

5. **Card shuffling:** A perfectly shuffled deck of 52 cards has $52!$ equally likely arrangements, so entropy is $\log_2(52!) \approx 225.58$ bits. Any deterministic shuffling algorithm that uses fewer than 225.58 random bits cannot produce a perfectly uniform distribution over all arrangements.

## AI/ML Relevance

**Decision trees:** Entropy is the most common splitting criterion in decision tree algorithms (ID3, C4.5). The algorithm selects the feature that maximises the reduction in entropy (information gain). Entropy-based splitting tends to produce balanced, shallow trees.

**Prediction uncertainty:** A classifier's softmax output can be interpreted as a probability distribution. The entropy of this distribution measures prediction uncertainty: low entropy means high confidence (one class dominates), high entropy means the model is uncertain (uniform across classes). Entropy-based confidence calibration helps detect out-of-distribution inputs.

**Maximum entropy models:** The MaxEnt principle states that among all distributions consistent with observed constraints, the one with maximum entropy should be chosen. This is the least informative distribution that satisfies the constraints — it makes no unwarranted assumptions. MaxEnt models are equivalent to logistic regression under certain conditions.

**Reinforcement learning:** Entropy regularisation adds an entropy bonus to the reward function, encouraging exploration by preventing the policy from becoming too deterministic too quickly. The Soft Actor-Critic (SAC) algorithm uses this principle, optimising $\mathbb{E}[R] + \alpha H(\pi(\cdot|s))$, where $\alpha$ controls exploration.

**Clustering evaluation:** The entropy of a clustering solution (given ground-truth labels) measures the purity of clusters. Lower cluster entropy indicates better alignment with true classes.

**Anomaly detection:** Data points with low probability under a fitted model have high surprise (information content $-\log P(x)$). Points with unusually high information content relative to the entropy of the distribution are flagged as anomalies.

## Mathematical Explanation

Shannon derived entropy from three axioms. Let $H(p_1, \dots, p_n)$ be a function measuring the uncertainty of a distribution with $n$ outcomes:

1. **Continuity:** $H$ is continuous in each $p_i$.
2. **Symmetry:** $H$ is symmetric (permuting probabilities does not change uncertainty).
3. **Maximum:** $H$ is maximised when all outcomes are equally likely.
4. **Additivity:** For independent events $A$ and $B$, $H(A, B) = H(A) + H(B)$. More generally, for a compound experiment: $H(X, Y) = H(X) + H(Y|X)$.

The unique function satisfying these properties is $H = -K \sum_{i} p_i \log p_i$, where $K$ is a positive constant (equivalently, the base of the logarithm).

For a continuous random variable with probability density $f(x)$, the **differential entropy** is:

$$
h(X) = -\int_{\mathcal{X}} f(x) \log f(x) \, dx
$$

Differential entropy inherits some but not all properties of discrete entropy — it can be negative, which makes it less interpretable. The relative entropy (KL divergence) between continuous distributions remains well-behaved.

**Joint entropy** of two discrete variables:

$$
H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log_2 P(x, y)
$$

**Conditional entropy**:

$$
H(Y|X) = \sum_{x \in \mathcal{X}} P(x) H(Y|X = x) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(y|x)
$$

The **chain rule** relates these: $H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$.

**Binary entropy function:** For a Bernoulli variable with $P(1) = p$, $H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$. This function is concave, symmetric about $p=0.5$, and attains its maximum of 1 bit at $p=0.5$.

## Formula(s)

**Shannon entropy (discrete, bits):**
$$
H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
$$

**Shannon entropy (discrete, nats):**
$$
H(X) = -\sum_{i=1}^{n} P(x_i) \ln P(x_i)
$$

**Binary entropy function:**
$$
H_2(p) = -p \log_2 p - (1-p) \log_2 (1-p)
$$

**Joint entropy:**
$$
H(X, Y) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(x, y)
$$

**Conditional entropy:**
$$
H(Y|X) = -\sum_{x} \sum_{y} P(x, y) \log_2 P(y|x)
$$

**Chain rule:**
$$
H(X, Y) = H(X) + H(Y|X)
$$

**Differential entropy (continuous):**
$$
h(X) = -\int_{\mathcal{X}} f(x) \log f(x) \, dx
$$

## Properties

- **Non-negativity:** $H(X) \geq 0$ for discrete $X$, with equality iff $X$ is deterministic ($P(x_i) = 1$ for some $i$).
- **Maximum entropy:** $H(X) \leq \log_2 |\mathcal{X}|$ for discrete $X$, with equality iff $X$ is uniform over $\mathcal{X}$.
- **Concavity:** $H$ is a concave function of the probability distribution. This means mixing distributions increases entropy: $H(\lambda P + (1-\lambda) Q) \geq \lambda H(P) + (1-\lambda) H(Q)$.
- **Invariance to outcome labelling:** Entropy depends only on the probability values, not on which outcome has which probability.
- **Additivity for independence:** If $X$ and $Y$ are independent, $H(X, Y) = H(X) + H(Y)$.
- **Conditioning reduces entropy:** $H(Y|X) \leq H(Y)$, with equality iff $X$ and $Y$ are independent. Knowing $X$ can only reduce (or maintain) uncertainty about $Y$.
- **Chain rule:** $H(X_1, \dots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, \dots, X_{i-1})$.
- **Data-processing inequality:** If $X \to Y \to Z$ is a Markov chain, then $H(Z) \geq H(Z|Y) \geq H(Z|X)$ and $H(Y) \geq H(Y|Z)$.

## Step-by-Step Worked Examples

### Example 1: Entropy of a Fair Coin

**Problem:** Compute the entropy of a fair coin flip in bits.

**Solution:**

Step 1: Identify the distribution. $P(H) = 0.5$, $P(T) = 0.5$.

Step 2: Apply the entropy formula.
$$
H = -[P(H) \log_2 P(H) + P(T) \log_2 P(T)]
$$

Step 3: Substitute values.
$$
H = -[0.5 \log_2 0.5 + 0.5 \log_2 0.5]
$$

Step 4: Simplify. $\log_2 0.5 = \log_2(1/2) = \log_2 1 - \log_2 2 = 0 - 1 = -1$.
$$
H = -[0.5(-1) + 0.5(-1)] = -[-0.5 - 0.5] = 1
$$

**Answer:** $H = 1$ bit.

Interpretation: Each flip conveys exactly 1 bit of information. An optimal code needs 1 bit per symbol.

### Example 2: Entropy of a Biased Coin

**Problem:** A coin lands heads with probability $p = 0.9$ and tails with probability $1-p = 0.1$. Compute its entropy. How does it compare to a fair coin?

**Solution:**

Step 1: Use the binary entropy function $H_2(p) = -p\log_2 p - (1-p)\log_2(1-p)$.

Step 2: Substitute $p = 0.9$.
$$
H = -0.9 \log_2 0.9 - 0.1 \log_2 0.1
$$

Step 3: Compute logs. $\log_2 0.9 = \frac{\ln 0.9}{\ln 2} \approx \frac{-0.10536}{0.69315} \approx -0.1520$. $\log_2 0.1 = \frac{\ln 0.1}{\ln 2} \approx \frac{-2.30259}{0.69315} \approx -3.3219$.

Step 4: Compute each term.
$$
-0.9(-0.1520) = 0.1368
$$
$$
-0.1(-3.3219) = 0.3322
$$

Step 5: Sum.
$$
H = 0.1368 + 0.3322 = 0.4690
$$

**Answer:** $H \approx 0.469$ bits. This is less than the fair coin's 1 bit because the outcome is more predictable.

Interpretation: On average, each flip conveys only 0.469 bits of information. An optimal code would use fewer than 1 bit per symbol on average.

### Example 3: Entropy of a Six-Sided Die

**Problem:** Compute the entropy of a fair six-sided die in both bits and nats.

**Solution:**

Step 1: Each face has probability $1/6$.

Step 2: Entropy in bits.
$$
H_{\text{bits}} = -\sum_{i=1}^{6} \frac{1}{6} \log_2 \frac{1}{6} = -6 \cdot \frac{1}{6} \log_2 \frac{1}{6} = -\log_2 \frac{1}{6} = \log_2 6
$$

Step 3: Compute $\log_2 6 = \frac{\ln 6}{\ln 2} \approx \frac{1.79176}{0.69315} \approx 2.5850$.

Step 4: Entropy in nats. Since $\ln 6 \approx 1.79176$, and using $-\sum \frac{1}{6} \ln \frac{1}{6} = \ln 6$:
$$
H_{\text{nats}} = \ln 6 \approx 1.7918
$$

**Answer:** $H_{\text{bits}} \approx 2.585$ bits, $H_{\text{nats}} \approx 1.792$ nats.

Interpretation: A fair six-sided die requires about 2.585 bits per roll for optimal encoding. Since 3 bits can represent 8 outcomes, a straightforward 3-bit encoding wastes about 0.415 bits per roll. Optimal coding (e.g., arithmetic coding) approaches 2.585 bits.

### Example 4: Entropy of a Non-Uniform Three-Outcome Distribution

**Problem:** A random variable $X$ takes values $\{A, B, C\}$ with $P(A) = 0.5$, $P(B) = 0.3$, $P(C) = 0.2$. Compute its entropy.

**Solution:**

Step 1: Apply the entropy formula.
$$
H = -[0.5\log_2 0.5 + 0.3\log_2 0.3 + 0.2\log_2 0.2]
$$

Step 2: Compute logs.
- $\log_2 0.5 = -1$
- $\log_2 0.3 = \frac{\ln 0.3}{\ln 2} \approx \frac{-1.20397}{0.69315} \approx -1.7370$
- $\log_2 0.2 = \frac{\ln 0.2}{\ln 2} \approx \frac{-1.60944}{0.69315} \approx -2.3219$

Step 3: Compute each term.
- $0.5 \times (-1) = -0.5$
- $0.3 \times (-1.7370) = -0.5211$
- $0.2 \times (-2.3219) = -0.4644$

Step 4: Sum and negate.
$$
H = -[(-0.5) + (-0.5211) + (-0.4644)] = -[-1.4855] = 1.4855
$$

**Answer:** $H \approx 1.486$ bits.

Check: For 3 equally likely outcomes, the maximum entropy would be $\log_2 3 \approx 1.585$ bits. Our answer is less, as expected for a non-uniform distribution.

## Visual Interpretation

Entropy can be visualised as a function of probability values. The binary entropy function $H_2(p)$ plotted against $p$ is a concave, symmetric curve peaking at $p=0.5$ with value 1, and dropping to 0 at $p=0$ and $p=1$.

For a distribution over $n$ outcomes, entropy is maximised at the uniform distribution — a flat, broad distribution. As the distribution becomes more peaked (one outcome dominates), entropy decreases. This can be visualised as the "spread" or "flatness" of the probability histogram.

A useful geometric interpretation: entropy is related to the volume of the typical set. For i.i.d. sequences of length $n$ drawn from $P$, there are approximately $2^{nH(X)}$ "typical" sequences, each with probability roughly $2^{-nH(X)}$. This is the asymptotic equipartition property (AEP). The typical set is the smallest set that contains most of the probability mass.

## Common Mistakes

1. **Using the wrong logarithm base:** Using $\log$ (base 10) instead of $\log_2$ (bits) or $\ln$ (nats). The numerical value depends on the base; always specify the base or unit.

2. **Forgetting $0\log 0 = 0$:** When a probability is zero, the term $0 \log 0$ is indeterminate. By convention and justified by continuity, it is treated as 0. Many beginners struggle with this.

3. **Confusing entropy with variance:** Both measure spread, but entropy measures uncertainty about the outcome of a random variable, while variance measures squared deviation from the mean. A bimodal distribution can have large variance but low entropy if it concentrates on two specific values.

4. **Assuming entropy is always bounded by 1:** The binary entropy function is bounded by 1, but entropy of an $n$-outcome distribution is bounded by $\log_2 n$, which can be arbitrarily large.

5. **Interpreting differential entropy the same way as discrete entropy:** Differential entropy can be negative (e.g., a uniform distribution on $[0, 0.5]$ has $h = -\ln 2 \approx -0.693$ nats). It does not have the same absolute interpretation as discrete entropy and is primarily used in relative comparisons.

6. **Confusing entropy with cross-entropy or KL divergence:** Entropy $H(P)$ measures uncertainty of $P$ itself. Cross-entropy $H(P, Q)$ measures the average number of bits needed to encode $P$ using $Q$'s code. KL divergence $D_{KL}(P\|Q) = H(P, Q) - H(P)$ measures the extra bits due to using $Q$ instead of $P$.

7. **Thinking entropy is a property of the event, not the distribution:** Entropy is a property of the entire probability distribution, not of individual outcomes. The information content of a single outcome ($-\log P(x_i)$) is called the surprisal or self-information, and its expectation is entropy.

## Interview Questions

### Beginner - 5

1. **Q:** What is Shannon entropy?
   **A:** Shannon entropy measures the average uncertainty or information content of a random variable's outcomes, defined as $H(X) = -\sum P(x_i)\log_2 P(x_i)$ in bits.

2. **Q:** What is the entropy of a fair coin flip?
   **A:** 1 bit, because $H = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1$.

3. **Q:** What is the entropy of a deterministic variable?
   **A:** 0, since there is no uncertainty — one outcome has probability 1 and all others 0.

4. **Q:** What distribution maximises entropy for a given set of outcomes?
   **A:** The uniform distribution, where all outcomes are equally likely.

5. **Q:** What is the difference between bits and nats?
   **A:** Bits use base-2 logarithms; nats use natural logarithms. $H_{\text{nats}} = H_{\text{bits}} \times \ln 2$.

### Intermediate - 5

1. **Q:** How does conditioning on another variable affect entropy?
   **A:** Conditioning reduces or maintains entropy: $H(Y|X) \leq H(Y)$. Knowing $X$ cannot increase uncertainty about $Y$.

2. **Q:** What is the relationship between joint entropy, marginal entropy, and conditional entropy?
   **A:** The chain rule: $H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$. This is analogous to the chain rule in probability.

3. **Q:** How is entropy used in decision tree learning?
   **A:** Decision trees use entropy to select splitting features. The feature that maximises information gain (reduction in entropy) is chosen at each node. This produces trees that minimise the expected number of questions needed to classify an instance.

4. **Q:** What is differential entropy and how does it differ from discrete entropy?
   **A:** Differential entropy extends entropy to continuous random variables: $h(X) = -\int f(x) \log f(x) dx$. Unlike discrete entropy, it can be negative and is not invariant under coordinate transformations. It is best interpreted relative to another distribution.

5. **Q:** Explain the maximum entropy principle with an example.
   **A:** The MaxEnt principle chooses the distribution with the largest entropy among those satisfying known constraints. For example, given only that a binary variable has mean 0.3, MaxEnt selects the Bernoulli(0.3) distribution — it makes no additional assumptions.

### Advanced - 3

1. **Q:** Prove that the uniform distribution maximises entropy over a finite set.
   **A:** Use Gibbs' inequality or the log-sum inequality. For any distribution $P$ over $n$ outcomes, $H(P) = -\sum p_i \log p_i \leq -\sum p_i \log (1/n) = \log n$. Equality holds when $p_i = 1/n$ for all $i$, i.e., the uniform distribution. Alternatively, use concavity of $H$ and Jensen's inequality.

2. **Q:** Derive the asymptotic equipartition property (AEP) for i.i.d. sequences and explain its significance.
   **A:** For i.i.d. $X_1, \dots, X_n \sim P$, the AEP states that $-\frac{1}{n}\log P(X_1, \dots, X_n) \xrightarrow{P} H(X)$. This means the set of sequences is approximately partitioned into a "typical set" of size $2^{nH(X)}$, each with probability approximately $2^{-nH(X)}$, and the remaining sequences have negligible total probability. This justifies entropy as the fundamental compression limit.

3. **Q:** Compare and contrast Shannon entropy with algorithmic (Kolmogorov) complexity. When would you use one over the other?
   **A:** Shannon entropy characterises the average information of a random source with known distribution. Kolmogorov complexity measures the minimum description length of an individual sequence without assuming a distribution. Shannon entropy applies when the data-generating process is probabilistic; Kolmogorov complexity applies to deterministic strings and is uncomputable but useful theoretically. They are related by the Levin-Schnorr theorem: a sequence is Martin-Lof random iff its Kolmogorov complexity is close to its Shannon entropy bound.

## Practice Problems

### Easy - 5

1. Compute the entropy of a Bernoulli random variable with $p = 0.25$ in bits.

2. A random variable has 4 equally likely outcomes. What is its entropy in bits?

3. True or false: $H(X) = 0$ if and only if $X$ is deterministic.

4. What is the maximum possible entropy of a random variable with 8 outcomes?

5. Convert 1 nat to bits.

### Medium - 5

1. A random variable has distribution $P = [0.4, 0.3, 0.2, 0.1]$. Compute its entropy.

2. Given $H(X) = 1.5$ bits and $H(Y) = 2$ bits with $H(X, Y) = 2.8$ bits, find $H(Y|X)$.

3. Show that $H(p) = H(1-p)$ for the binary entropy function.

4. A classifier outputs probabilities $[0.8, 0.15, 0.05]$ for three classes. Compute the entropy of this prediction.

5. For two independent random variables $X$ and $Y$, prove $H(X, Y) = H(X) + H(Y)$.

### Hard - 3

1. Prove that $H(X) \geq H(X|Y)$ for any jointly distributed $X$ and $Y$. Under what conditions does equality hold?

2. Derive the maximum entropy distribution for a continuous random variable with known mean $\mu$ and variance $\sigma^2$ on the real line. What family of distributions emerges?

3. Prove the entropy chain rule: $H(X_1, X_2, \dots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, \dots, X_{i-1})$.

## Solutions

**Easy:**

1. $H = -0.25\log_2 0.25 - 0.75\log_2 0.75 = -0.25(-2) - 0.75(-0.4150) = 0.5 + 0.3113 = 0.8113$ bits.

2. $H = \log_2 4 = 2$ bits (uniform distribution).

3. True. If $P(x_i) = 1$ for some $i$, then $-\sum P(x_i)\log P(x_i) = -1\log 1 - 0\log 0 = 0$.

4. $H_{\max} = \log_2 8 = 3$ bits.

5. $1$ nat $= 1/\ln 2 \approx 1.4427$ bits.

**Medium:**

1. $H = -[0.4\log_2 0.4 + 0.3\log_2 0.3 + 0.2\log_2 0.2 + 0.1\log_2 0.1] = -[0.4(-1.3219) + 0.3(-1.7370) + 0.2(-2.3219) + 0.1(-3.3219)] = -[-0.5288 - 0.5211 - 0.4644 - 0.3322] = 1.8464$ bits.

2. $H(Y|X) = H(X,Y) - H(X) = 2.8 - 1.5 = 1.3$ bits.

3. $H(p) = -p\log p - (1-p)\log(1-p)$. $H(1-p) = -(1-p)\log(1-p) - p\log p = H(p)$. The function is symmetric about $p=0.5$.

4. $H = -[0.8\log_2 0.8 + 0.15\log_2 0.15 + 0.05\log_2 0.05] = -[0.8(-0.3219) + 0.15(-2.7370) + 0.05(-4.3219)] = -[-0.2575 - 0.4106 - 0.2161] = 0.8842$ bits. This relatively low entropy indicates high confidence.

5. $H(X,Y) = -\sum_x\sum_y P(x)P(y) \log(P(x)P(y)) = -\sum_x\sum_y P(x)P(y)[\log P(x) + \log P(y)] = -\sum_x P(x)\log P(x) \sum_y P(y) - \sum_y P(y)\log P(y) \sum_x P(x) = H(X) \cdot 1 + H(Y) \cdot 1 = H(X) + H(Y)$.

**Hard:**

1. $H(X) - H(X|Y) = I(X;Y) \geq 0$ by the non-negativity of KL divergence (mutual information is $D_{KL}(P(x,y)\|P(x)P(y))$). Equality holds iff $X$ and $Y$ are independent, i.e., $P(x,y) = P(x)P(y)$.

2. Maximising $h(f) = -\int f(x) \ln f(x) dx$ subject to $\int f(x) dx = 1$, $\int x f(x) dx = \mu$, $\int (x-\mu)^2 f(x) dx = \sigma^2$ using variational calculus gives $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$, the Gaussian distribution. The normal distribution is the maximum entropy distribution for given mean and variance.

3. By repeated application of the chain rule for two variables: $H(X_1, \dots, X_n) = H(X_1) + H(X_2|X_1) + H(X_3|X_1, X_2) + \cdots + H(X_n|X_1, \dots, X_{n-1})$. This follows from the definition of conditional entropy and the chain rule $P(x_1, \dots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, \dots, x_{i-1})$.

## Related Concepts

- Cross Entropy (MATH-089) — average number of bits to encode $P$ using $Q$'s code
- KL Divergence (MATH-090) — $D_{KL}(P\|Q) = H(P,Q) - H(P)$, the extra bits from using $Q$
- Mutual Information (MATH-091) — $I(X;Y) = H(X) - H(X|Y)$, shared information between variables
- Information Gain (MATH-092) — reduction in entropy from splitting on a feature
- Conditional Entropy — uncertainty in $Y$ given $X$
- Joint Entropy — total uncertainty in a pair of variables
- Binary Entropy Function — $H_2(p)$ for Bernoulli distributions
- Asymptotic Equipartition Property — convergence of log-probability to entropy

## Next Concepts

- Cross Entropy (MATH-089) — loss function for classification
- KL Divergence (MATH-090) — measuring distribution divergence
- Mutual Information (MATH-091) — dependence measure
- Information Gain (MATH-092) — decision tree splitting criterion
- Channel Capacity — maximum rate of reliable communication

## Summary

Entropy, introduced by Claude Shannon in 1948, is the foundational measure of uncertainty in information theory. For a discrete random variable $X$, entropy is $H(X) = -\sum P(x_i)\log_2 P(x_i)$, measured in bits. It is non-negative, maximised by the uniform distribution, and additive for independent variables. Entropy quantifies the average information content of a random source and sets the fundamental limit for lossless compression. In machine learning, entropy appears in decision tree splitting, prediction uncertainty quantification, maximum entropy modelling, and reinforcement learning regularisation. The binary entropy function $H_2(p) = -p\log p - (1-p)\log(1-p)$ is a special case for two-outcome distributions.

## Key Takeaways

- $H(X) = -\sum P(x)\log_2 P(x)$ measures average uncertainty in bits (or nats with $\ln$).
- $0 \leq H(X) \leq \log_2 |\mathcal{X}|$; lower bound at determinism, upper bound at uniformity.
- $\log_2$ gives bits; $\ln$ gives nats; $1$ bit $= \ln 2$ nats.
- Conditioning reduces entropy: $H(Y|X) \leq H(Y)$.
- Binary entropy $H_2(p)$ is concave, symmetric about $p=0.5$, max 1 at $p=0.5$.
- Entropy is not variance; it measures uncertainty about outcomes, not numerical spread.
- In ML: decision tree splitting, confidence calibration, MaxEnt models, RL exploration.
- Foundation for cross-entropy, KL divergence, mutual information, and information gain.
