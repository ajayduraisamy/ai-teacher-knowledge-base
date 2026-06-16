# Concept: Random Variable

## Concept ID

MATH-070

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- Define a random variable as a function from sample space to real numbers
- Distinguish between discrete and continuous random variables
- Define and compute probability mass functions (PMF) and probability density functions (PDF)
- Define and interpret cumulative distribution functions (CDF)
- Compute expected value and variance of random variables
- Connect random variables to stochastic processes in machine learning

## Prerequisites

- Probability concepts (MATH-065)
- Sample space concept (MATH-066)
- Basic calculus (integration and summation)
- Event operations (MATH-067)

## Definition

A **random variable** is a function that maps outcomes from a sample space to real numbers. Formally, a random variable X is a function X: S -> R where S is the sample space. For each outcome w in S, X(w) is a real number.

Random variables are classified as:

**Discrete Random Variable**: Takes on a countable number of values (finite or countably infinite). Described by a probability mass function (PMF).

**Continuous Random Variable**: Takes on uncountably many values (typically an interval of real numbers). Described by a probability density function (PDF).

## Intuition

A random variable turns random outcomes into numbers. Instead of saying "the coin shows heads," we say X = 1 for heads and X = 0 for tails. This numerical representation lets us use mathematics to compute averages, variances, and probabilities.

Think of a random variable as a measurement you take on a random experiment. The experiment produces an outcome, and you measure some numerical property of that outcome. The measurement value is random because the outcome is random.

## Why This Concept Matters

Random variables are the language of data science and machine learning. Every dataset consists of observed values of random variables. Every model defines relationships between random variables. Expected values (means) appear in every loss function. Variance measures uncertainty. The concept of a random variable bridges probability theory and statistics, enabling us to analyse data probabilistically, build predictive models, and quantify uncertainty.

## Historical Background

The formal definition of a random variable as a measurable function emerged in the early 20th century with the development of measure-theoretic probability. **Andrey Kolmogorov** (1933) provided the rigorous foundation in his axiomatisation. The term "random variable" was introduced earlier by **Vladimir Markov** (1856-1922) and popularised by **Ronald Fisher** (1890-1962). The distinction between discrete and continuous random variables was understood by **Pierre-Simon Laplace** and **Carl Friedrich Gauss** in the context of error analysis, but the unified treatment using measure theory is due to Kolmogorov and **Henri Lebesgue**.

## Real World Examples

1. **Number of Website Visitors** - The number of visitors to a website in an hour is a discrete random variable (count). It follows a Poisson distribution if visitors arrive independently at a constant average rate.

2. **Height of a Random Person** - Height is a continuous random variable measured in centimetres or inches. It approximately follows a normal distribution.

3. **Stock Return** - The daily return of a stock is a continuous random variable. Financial models treat returns as random variables with specific distributions (often assumed normal, though empirically they have heavy tails).

4. **Number of Defective Items** - In a batch of 100 manufactured items, the number of defective items is a discrete random variable following a hypergeometric or binomial distribution.

5. **Time Until System Failure** - The time until a server crashes is a continuous random variable, often modelled with an exponential or Weibull distribution in reliability engineering.

## AI/ML Relevance

1. **Dropout as a Random Variable** - Dropout regularisation multiplies each neuron's output by a Bernoulli random variable Z ~ Bernoulli(p). During training, Z = 1 with probability p (neuron active) and Z = 0 with probability 1-p (neuron dropped). This makes the network's behaviour stochastic, preventing co-adaptation of neurons.

2. **Data Generation as Random Process** - Observed data is assumed to be generated from an underlying random process. Each data point is a realisation of a random variable X. The goal of generative modelling is to learn the distribution P(X) or P(X|Y).

3. **Stochastic Gradient Descent** - The gradient estimate in SGD is a random variable because it is computed on a random mini-batch. The noise in this random variable helps escape local minima and generalise better.

4. **Latent Variables** - In variational autoencoders and other latent variable models, the latent code z is a random variable. The encoder approximates P(z|x) and the decoder models P(x|z). Random sampling from the latent distribution enables generation of new data.

5. **Reinforcement Learning Value Functions** - The return (cumulative reward) in RL is a random variable because it depends on the stochastic policy and environment dynamics. Value functions are expectations of this random variable.

6. **Monte Carlo Methods** - Monte Carlo estimation uses random sampling to approximate expectations of random variables. The estimator itself is a random variable whose variance decreases with sample size.

## Mathematical Explanation

### Discrete Random Variables

For a discrete random variable X with values {x_1, x_2, ...}, the probability mass function (PMF) is:
p_X(x) = P(X = x)

Properties of the PMF:
- p_X(x) >= 0 for all x
- sum_i p_X(x_i) = 1
- P(X in A) = sum_{x in A} p_X(x)

The cumulative distribution function (CDF) is:
F_X(x) = P(X <= x) = sum_{x_i <= x} p_X(x_i)

### Continuous Random Variables

For a continuous random variable X, the probability density function (PDF) f_X(x) satisfies:
P(a <= X <= b) = int_a^b f_X(x) dx

Properties of the PDF:
- f_X(x) >= 0 for all x
- int_{-inf}^{inf} f_X(x) dx = 1
- P(X = a) = 0 for any specific value a

The CDF for continuous X is:
F_X(x) = P(X <= x) = int_{-inf}^x f_X(t) dt

The PDF is the derivative of the CDF: f_X(x) = d/dx F_X(x).

### Expected Value

**Discrete**: E[X] = sum_i x_i P(X = x_i)

**Continuous**: E[X] = int_{-inf}^{inf} x f_X(x) dx

**Linearity**: E[aX + bY] = a E[X] + b E[Y]

### Variance

Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2

Standard deviation: SD(X) = sqrt(Var(X))

### Functions of Random Variables

If Y = g(X) is a function of random variable X:
- Discrete: E[Y] = sum_i g(x_i) P(X = x_i)
- Continuous: E[Y] = int_{-inf}^{inf} g(x) f_X(x) dx

## Formula(s)

1. **PMF (Discrete)**:
p_X(x) = P(X = x), sum_x p_X(x) = 1

2. **PDF (Continuous)**:
P(a <= X <= b) = int_a^b f_X(x) dx, int_{-inf}^{inf} f_X(x) dx = 1

3. **CDF**:
F_X(x) = P(X <= x)

4. **Expected Value**:
E[X] = sum_x x p_X(x) [discrete] or int x f_X(x) dx [continuous]

5. **Variance**:
Var(X) = E[X^2] - (E[X])^2

6. **Linearity of Expectation**:
E[aX + bY] = a E[X] + b E[Y]

7. **Law of the Unconscious Statistician**:
E[g(X)] = sum_x g(x) p_X(x) [discrete] or int g(x) f_X(x) dx [continuous]

## Properties

1. **Non-negativity of PMF/PDF**: p_X(x) >= 0 and f_X(x) >= 0.
2. **Normalisation**: Sum or integral of PMF/PDF equals 1.
3. **CDF Properties**: F_X is non-decreasing, right-continuous, with limit 0 as x->-inf and 1 as x->+inf.
4. **Linearity of Expectation**: E[aX + bY] = aE[X] + bE[Y] (no independence required).
5. **Variance Scaling**: Var(aX + b) = a^2 Var(X).
6. **Non-negative Variance**: Var(X) >= 0, with equality iff X is constant.
7. **Chebyshev's Inequality**: P(|X - mu| >= k sigma) <= 1/k^2 for any k > 0.
8. **Markov's Inequality**: For non-negative X, P(X >= a) <= E[X]/a.

## Step-by-Step Worked Examples

### Example 1: Expected Value of a Discrete Random Variable

**Problem**: Let X be the sum of two fair six-sided dice. Compute the PMF and expected value of X.

**Solution**:

Step 1: Determine the sample space. Sum ranges from 2 to 12.

Step 2: Count outcomes for each sum:
| x | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| Count | 1 | 2 | 3 | 4 | 5 | 6 | 5 | 4 | 3 | 2 | 1 |

Step 3: Compute PMF: P(X = x) = count / 36.
P(2) = 1/36, P(3) = 1/18, P(4) = 1/12, P(5) = 1/9, P(6) = 5/36, P(7) = 1/6, P(8) = 5/36, P(9) = 1/9, P(10) = 1/12, P(11) = 1/18, P(12) = 1/36.

Step 4: Verify sum: (1+2+3+4+5+6+5+4+3+2+1)/36 = 1.

Step 5: Compute E[X]:
E[X] = 2(1/36) + 3(2/36) + 4(3/36) + 5(4/36) + 6(5/36) + 7(6/36) + 8(5/36) + 9(4/36) + 10(3/36) + 11(2/36) + 12(1/36)
E[X] = (2+6+12+20+30+42+40+36+30+22+12)/36 = 252/36 = 7.

Step 6: Interpret. The expected sum of two dice is 7.

### Example 2: Continuous Random Variable and Expected Value

**Problem**: Let X be a continuous random variable with PDF f_X(x) = 2x for 0 <= x <= 1, and 0 otherwise. Compute the CDF and E[X].

**Solution**:

Step 1: Verify the PDF integrates to 1:
int_0^1 2x dx = [x^2]_0^1 = 1. Correct.

Step 2: Compute the CDF:
F_X(x) = int_0^x 2t dt = t^2 evaluated from 0 to x = x^2, for 0 <= x <= 1.
F_X(x) = 0 for x < 0, F_X(x) = 1 for x > 1.

Step 3: Compute E[X]:
E[X] = int_0^1 x * 2x dx = int_0^1 2x^2 dx = [2x^3/3]_0^1 = 2/3.

Step 4: Interpret. The expected value is 2/3, which is not the midpoint (0.5) because the PDF is skewed right.

### Example 3: Variance of a Random Variable

**Problem**: Let X represent the number of heads in 3 fair coin flips. Compute Var(X).

**Solution**:

Step 1: X follows Binomial(3, 0.5).

Step 2: PMF: P(X=0) = 1/8, P(X=1) = 3/8, P(X=2) = 3/8, P(X=3) = 1/8.

Step 3: Compute E[X]:
E[X] = 0(1/8) + 1(3/8) + 2(3/8) + 3(1/8) = 12/8 = 1.5.

Step 4: Compute E[X^2]:
E[X^2] = 0(1/8) + 1(3/8) + 4(3/8) + 9(1/8) = 24/8 = 3.

Step 5: Compute Var(X) = E[X^2] - (E[X])^2 = 3 - 2.25 = 0.75.

Step 6: Standard deviation: SD(X) = sqrt(0.75) approx 0.866.

Step 7: Alternative using known formula: For Binomial(n,p), Var(X) = np(1-p) = 3*0.5*0.5 = 0.75. Confirms.

### Example 4: Function of a Random Variable

**Problem**: Let X be a continuous random variable with PDF f_X(x) = 1 for 0 <= x <= 1 (uniform). Let Y = X^2. Find E[Y].

**Solution**:

Step 1: Using the Law of the Unconscious Statistician (LOTUS):
E[Y] = E[X^2] = int_0^1 x^2 * 1 dx = [x^3/3]_0^1 = 1/3.

Step 2: Alternatively, find the distribution of Y and compute directly. For y in [0,1]:
F_Y(y) = P(Y <= y) = P(X^2 <= y) = P(X <= sqrt(y)) = sqrt(y).
f_Y(y) = d/dy sqrt(y) = 1/(2 sqrt(y)) for 0 <= y <= 1.

Step 3: Compute E[Y] using f_Y:
E[Y] = int_0^1 y * 1/(2 sqrt(y)) dy = int_0^1 sqrt(y)/2 dy = [y^{3/2}/3]_0^1 = 1/3.

Step 4: Both methods give the same result, confirming the LOTUS theorem.

### Example 5: Linear Combination of Random Variables

**Problem**: Two independent random variables: E[X] = 3, Var(X) = 2, E[Y] = -1, Var(Y) = 4. Let Z = 2X - 3Y + 5. Find E[Z] and Var(Z).

**Solution**:

Step 1: Compute E[Z] using linearity:
E[Z] = 2E[X] - 3E[Y] + 5 = 2*3 - 3*(-1) + 5 = 6 + 3 + 5 = 14.

Step 2: Compute Var(Z). For independent X and Y:
Var(Z) = Var(2X - 3Y + 5) = 2^2 Var(X) + (-3)^2 Var(Y) = 4*2 + 9*4 = 8 + 36 = 44.

Step 3: The constant 5 disappears (Var(aX + b) = a^2 Var(X)).

Step 4: Standard deviation: SD(Z) = sqrt(44) approx 6.633.

## Visual Interpretation

The PMF of a discrete random variable is visualised as a bar chart or stem plot. Each value x has a vertical line or bar at height p_X(x). The sum of all bar heights equals 1.

The PDF of a continuous random variable is a smooth curve. The probability that X falls in an interval is the area under the curve over that interval. The total area under the PDF is 1.

The CDF is a non-decreasing function that starts at 0 and reaches 1. For a discrete random variable, the CDF is a step function. For a continuous random variable, the CDF is continuous and differentiable (almost everywhere).

The expected value is the centre of mass of the distribution - the point where the probability mass would balance if placed on a fulcrum.

## Common Mistakes

1. **Confusing PMF with PDF**: The PMF gives P(X = x) directly. The PDF at point x does NOT give P(X = x) (which is 0 for continuous variables). Only the integral of the PDF over an interval gives probability.

2. **Treating expectation as the most likely value**: E[X] is the long-run average, not the mode. For a skewed distribution, the mean can differ substantially from the most likely value.

3. **Assuming Var(X+Y) = Var(X) + Var(Y) always**: This holds only when X and Y are uncorrelated (or independent). In general, Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y).

4. **Forgetting the constant in variance**: Var(aX + b) = a^2 Var(X). The constant b does not affect variance, but the scaling factor a is squared.

5. **Applying discrete formulas to continuous variables**: Summing over all values for a continuous variable gives nonsense. Use integration for continuous variables.

6. **Confusing a random variable with its distribution**: The random variable X is the function mapping outcomes to numbers. Its distribution is the description of how probability is spread over those numbers.

7. **Ignoring the support**: The PMF or PDF is zero outside the support. Forgetting to specify the support leads to incorrect calculations.

## Interview Questions

### Beginner

1. **Q**: What is a random variable?
   **A**: A random variable is a function that maps outcomes from a sample space to real numbers.

2. **Q**: What is the difference between a discrete and continuous random variable?
   **A**: A discrete random variable takes countably many values (like integers) and has a PMF. A continuous random variable takes uncountably many values (like real numbers in an interval) and has a PDF.

3. **Q**: What is the expected value of a random variable?
   **A**: The expected value is the probability-weighted average of its possible values.

4. **Q**: If a fair coin is flipped, let X = 1 for heads and X = 0 for tails. What is E[X]?
   **A**: E[X] = 1*0.5 + 0*0.5 = 0.5.

5. **Q**: What does Var(X) measure?
   **A**: Variance measures the spread or dispersion of a random variable around its mean.

### Intermediate

1. **Q**: State the law of total expectation. Provide an intuitive explanation.
   **A**: The law of total expectation states E[X] = E[E[X|Y]]. It means the expected value of X can be computed by averaging the conditional expectation of X given Y over the distribution of Y.

2. **Q**: In machine learning, how is dropout implemented as a random variable?
   **A**: During training, each neuron output h_i is multiplied by a Bernoulli random variable Z_i ~ Bernoulli(p): h_i' = h_i * Z_i / p. The division by p preserves the expected value.

3. **Q**: Explain the bias-variance tradeoff in terms of random variables.
   **A**: For an estimator theta_hat of parameter theta, MSE = Bias(theta_hat)^2 + Var(theta_hat). Bias is systematic error, variance is random fluctuation.

4. **Q**: What is the CDF and how does it relate to the PDF?
   **A**: The CDF F_X(x) = P(X <= x). For continuous variables, the PDF is the derivative of the CDF: f_X(x) = d/dx F_X(x).

5. **Q**: Can a random variable have a negative expected value? Give an example.
   **A**: Yes. A gambling game that pays $5 with probability 0.1 and -$1 with probability 0.9 has E[X] = -0.4.

### Advanced

1. **Q**: Prove Var(X) = E[X^2] - (E[X])^2.
   **A**: Var(X) = E[(X-mu)^2] = E[X^2 - 2mu X + mu^2] = E[X^2] - 2mu E[X] + mu^2 = E[X^2] - mu^2 = E[X^2] - (E[X])^2.

2. **Q**: Derive the expectation and variance of a Bernoulli random variable.
   **A**: For X ~ Bernoulli(p): P(X=1)=p, P(X=0)=1-p. E[X] = p. E[X^2] = p. Var(X) = p - p^2 = p(1-p).

3. **Q**: Explain the role of random variables in the reparameterisation trick used in VAEs.
   **A**: The reparameterisation trick expresses z as a deterministic function of a standard random variable epsilon: z = mu(x) + sigma(x)*epsilon, where epsilon ~ N(0,I). This allows gradients to flow through mu and sigma.

## Practice Problems

### Easy

1. Let X be the number when a fair die is rolled. Find E[X].

2. A fair coin is flipped. Let X = 1 if heads, X = -1 if tails. Find E[X] and Var(X).

3. Let X have PMF: P(X=0)=0.2, P(X=1)=0.5, P(X=2)=0.3. Find E[X] and E[X^2].

4. For a continuous random variable with PDF f(x)=3x^2 for 0<=x<=1, find P(0.2<=X<=0.5).

5. If E[X]=2 and Var(X)=3, find E[3X-1] and Var(3X-1).

### Medium

1. Let X be Poisson with mean lambda. Show that E[X] = lambda.

2. X is a continuous random variable with PDF f(x)=cx^2 for 0<=x<=2. Find c and compute P(X>1).

3. Two independent random variables: E[X]=4, Var(X)=1, E[Y]=-2, Var(Y)=9. Find E[2X-Y+3] and Var(2X-Y+3).

4. Let X ~ Uniform(0,10). Find E[X], Var(X), and P(X>7|X>3).

5. A random variable X has CDF F(x)=0 for x<0, F(x)=x^2/4 for 0<=x<=2, F(x)=1 for x>2. Find the PDF and compute P(0.5<=X<=1.5).

### Hard

1. Prove that for any random variable X with finite variance, Var(X) = E[X^2] - (E[X])^2 and show Var(X) >= 0.

2. Let X be a continuous random variable with PDF f_X(x)=2(1-x) for 0<=x<=1. Find the PDF of Y = X^2.

3. Derive the expectation and variance of the sample mean bar{X} = (1/n) sum X_i where X_i are i.i.d. with mean mu and variance sigma^2.

## Solutions

### Easy Solutions

**Solution 1**: E[X] = (1+2+3+4+5+6)/6 = 3.5.

**Solution 2**: E[X] = 0. E[X^2] = 1. Var(X) = 1 - 0 = 1.

**Solution 3**: E[X] = 1.1. E[X^2] = 1.7.

**Solution 4**: P(0.2<=X<=0.5) = int_{0.2}^{0.5} 3x^2 dx = [x^3]_{0.2}^{0.5} = 0.125-0.008 = 0.117.

**Solution 5**: E[3X-1] = 3*2-1 = 5. Var(3X-1) = 9*3 = 27.

### Medium Solutions

**Solution 1**: E[X] = sum_{k=0}^{inf} k lambda^k e^{-lambda}/k! = lambda.

**Solution 2**: c=3/8. P(X>1) = 7/8.

**Solution 3**: E[2X-Y+3] = 13. Var(2X-Y+3) = 13.

**Solution 4**: E[X]=5, Var(X)=25/3. P(X>7|X>3) = 3/7.

**Solution 5**: f(x)=x/2 for 0<=x<=2. P(0.5<=X<=1.5) = 0.5.

### Hard Solutions

**Solution 1**: Var(X) = E[(X-mu)^2] = E[X^2-2mu X+mu^2] = E[X^2] - 2mu^2 + mu^2 = E[X^2] - mu^2. Since (X-mu)^2 >= 0, its expectation >=0.

**Solution 2**: For Y=X^2 on [0,1]: F_Y(y) = 2sqrt(y)-y. f_Y(y) = 1/sqrt(y)-1 for 0<=y<=1.

**Solution 3**: E[bar{X}] = mu. Var(bar{X}) = sigma^2/n.

## Related Concepts

- **Probability Distribution (MATH-071)**: A complete description of a random variable's probabilities
- **Probability (MATH-065)**: The foundation of random variable theory
- **Sample Space (MATH-066)**: The domain of a random variable
- **Conditional Probability (MATH-068)**: Conditional distributions of random variables

## Next Concepts

- **Probability Distribution**: Detailed study of specific distributions (MATH-071)
- **Joint Distributions**: Multiple random variables together
- **Central Limit Theorem**: Sums of random variables approach normality
- **Law of Large Numbers**: Sample means converge to expectations

## Summary

A random variable is a numerical summary of a random experiment. Discrete random variables take countable values (PMF), continuous take uncountable values (PDF). The CDF F(x)=P(X<=x) characterises any random variable. The expected value is the probability-weighted average, and variance measures spread. Random variables are fundamental to ML: they appear in stochastic gradient descent, dropout, latent variable models, and Monte Carlo methods.

## Key Takeaways

- A random variable maps outcomes to real numbers
- Discrete: PMF p(x)=P(X=x); Continuous: PDF f(x) with integral giving probabilities
- CDF F(x)=P(X<=x) fully characterises the distribution
- E[X] is the probability-weighted average (centre of mass)
- Var(X)=E[X^2]-(E[X])^2 measures spread
- Linearity of expectation: E[aX+bY]=aE[X]+bE[Y]
- Variance scales quadratically: Var(aX+b)=a^2 Var(X)
- The PDF is the derivative of the CDF for continuous variables
- LOTUS: E[g(X)] can be computed without finding the distribution of g(X)
- Random variables are central to ML: dropout, SGD noise, latent variables
