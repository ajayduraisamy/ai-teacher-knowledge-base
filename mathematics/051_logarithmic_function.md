# Concept: Logarithmic Function

## Concept ID

MATH-051

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define the logarithmic function $f(x) = \log_a(x)$ as the inverse of the exponential function $a^x$.
- Apply the laws of logarithms: $\log(ab) = \log a + \log b$, $\log(a^b) = b \log a$, $\log(a/b) = \log a - \log b$.
- Use the change of base formula to evaluate logarithms with arbitrary bases.
- Distinguish between common logarithm ($\log_{10}$) and natural logarithm ($\ln$).
- Solve logarithmic and exponential equations using logarithmic properties.
- Connect logarithms to AI/ML concepts: cross-entropy loss, log-likelihood, log-odds, and the log-sum-exp trick.

## Prerequisites

- Understanding of functions (MATH-044), domain (MATH-045), range (MATH-046), and inverse functions (MATH-048).
- Exponential functions (MATH-050): the logarithmic function is the inverse of the exponential.
- Basic exponent rules: $a^x a^y = a^{x+y}$, $(a^x)^y = a^{xy}$.
- Solving basic algebraic equations.

## Definition

The **logarithmic function** with base $a$ (where $a > 0$, $a \neq 1$) is defined as the inverse of the exponential function $a^x$:

$$f(x) = \log_a(x) \quad \text{if and only if} \quad a^{f(x)} = x$$

In words: $\log_a(x)$ is the exponent to which the base $a$ must be raised to produce $x$. For example, $\log_2(8) = 3$ because $2^3 = 8$.

The domain of $\log_a(x)$ is $(0, \infty)$ (positive real numbers), and the range is $\mathbb{R}$ (all real numbers). This swaps the domain and range of the exponential function $a^x$, which has domain $\mathbb{R}$ and range $(0, \infty)$.

Two bases are particularly important:
- **Common logarithm:** $\log_{10}(x)$ or simply $\log(x)$ (base 10).
- **Natural logarithm:** $\log_e(x)$ or $\ln(x)$ (base $e \approx 2.71828$).

The natural logarithm is the most important in mathematics and machine learning because of its natural relationship with $e^x$ and its appearance in calculus (derivatives and integrals).

## Intuition

A logarithm answers the question: "What exponent do I need?" If exponential functions ask "What is $2^5$?" (answer: 32), then logarithmic functions ask "To what power must I raise 2 to get 32?" (answer: 5, because $2^5 = 32$).

Think of a logarithm as an **exponent finder**. Given a base and a desired result, the logarithm tells you the exponent.

The logarithmic transformation has a remarkable effect: it **compresses multiplicative relationships into additive ones**. This is why logarithms are everywhere in science:
- Earthquake magnitudes (Richter scale) are logarithmic.
- Sound loudness (decibels) is logarithmic.
- Star brightness (magnitude scale) is logarithmic.
- pH (acidity) is logarithmic.

When you take a logarithm, multiplication becomes addition, division becomes subtraction, and exponentiation becomes multiplication. This turns difficult problems into simpler ones.

The natural logarithm $\ln x$ has a special geometric interpretation: it is the area under the curve $y = 1/t$ from $t = 1$ to $t = x$:

$$\ln x = \int_1^x \frac{1}{t} \, dt$$

For $x > 1$, this is a positive area. For $0 < x < 1$, it is a negative area (integral from $x$ to $1$).

## Why This Concept Matters

Logarithms are indispensable across science, engineering, and machine learning:

1. **Solving Exponential Equations:** Logarithms are the primary tool for solving equations where the variable is in the exponent. For example, $2^x = 50 \implies x = \log_2 50$.

2. **Data Transformation:** Logarithmic transformation converts multiplicative relationships into additive ones, making data more amenable to linear models. This is why economists often model $\ln(\text{GDP})$ rather than raw GDP.

3. **Scaling:** Logarithms handle numbers that span many orders of magnitude. The Richter scale, decibel scale, and pH scale are all logarithmic because they compress vast ranges into manageable numbers.

4. **Information Theory:** Shannon entropy $H = -\sum p_i \log_2 p_i$ measures information content in bits. The logarithm base 2 gives the number of binary digits needed to encode a message.

5. **Machine Learning:** Logarithms appear in loss functions (cross-entropy), likelihood estimation, probability modeling (log-odds), and numerical stability techniques (log-sum-exp).

## Historical Background

The Scottish mathematician John Napier (1550-1617) invented logarithms in the early 17th century as a computational aid. Before electronic calculators, multiplication and division were time-consuming. Napier's logarithms allowed these operations to be replaced by addition and subtraction using logarithm tables.

The key insight: if you have a table of logarithms, then:
- To compute $a \times b$: look up $\log a$ and $\log b$, add them, then find the anti-logarithm of the sum.
- To compute $a / b$: look up $\log a$ and $\log b$, subtract, then find the anti-logarithm.

Henry Briggs, a contemporary of Napier, developed the common logarithm (base 10) in 1617. Logarithm tables became essential tools for scientists and engineers for over 300 years, until the advent of electronic calculators.

The natural logarithm was developed by John Speidell (1619) and later formalized by Nicolaus Mercator (1668). The notation $\ln$ became standard in the 20th century.

Euler connected logarithms to exponentials rigorously in the 18th century, establishing $\ln x$ as the inverse of $e^x$. He also discovered the remarkable result $\ln(-1) = i\pi$ (already implicit in Euler's formula).

The slide rule, invented by William Oughtred in 1622, was a mechanical analog computer that used logarithmic scales. It remained the primary calculation tool for engineers until the 1970s.

## Real World Examples

**Example 1: Earthquake Magnitude (Richter Scale).** The Richter scale is logarithmic: each whole number increase corresponds to a tenfold increase in amplitude. A magnitude 6 earthquake is $10^{6-5} = 10$ times more powerful than magnitude 5, and $10^{6-4} = 100$ times more powerful than magnitude 4. The energy released increases by about $10^{1.5} \approx 31.6$ times per unit.

**Example 2: Sound Loudness (Decibels).** Decibels measure sound intensity logarithmically:
$$L = 10 \log_{10}\left(\frac{I}{I_0}\right)$$
where $I_0 = 10^{-12}$ W/m$^2$ is the threshold of hearing. A whisper at 20 dB has intensity $10^{2} I_0 = 10^{-10}$ W/m$^2$. A rock concert at 120 dB has intensity $10^{12} I_0 = 1$ W/m$^2$ — 10 billion times more intense.

**Example 3: pH Chemistry.** The pH of a solution measures hydrogen ion concentration $[H^+]$ logarithmically:
$$\text{pH} = -\log_{10}[H^+]$$
Pure water has $[H^+] = 10^{-7}$ M, giving pH = 7. Lemon juice has pH $\approx 2$, meaning $[H^+] \approx 10^{-2} = 0.01$ M — 100,000 times more acidic than water.

**Example 4: Population Growth.** The time for a population to double can be found using logarithms. If $P(t) = P_0 e^{rt}$, then $P_0 e^{rt} = 2P_0 \implies e^{rt} = 2 \implies rt = \ln 2 \implies t = \frac{\ln 2}{r}$. For $r = 0.02$ (2% growth), doubling time is $\ln 2 / 0.02 \approx 34.7$ years.

**Example 5: Moore's Law.** The number of transistors on a chip doubles approximately every 2 years. This exponential growth means that the log of transistor count increases linearly with time. Plotting $\log(\text{transistors})$ vs. year gives a straight line, making predictions simple.

## AI/ML Relevance

Logarithms are deeply embedded in machine learning, primarily through loss functions, probability, and numerical techniques.

**1. Cross-Entropy Loss.** For classification tasks, the cross-entropy loss is:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$$
where $y_{i,c}$ is the true label (one-hot encoded) and $\hat{y}_{i,c}$ is the predicted probability. The logarithm is crucial because:
- It heavily penalizes confident wrong predictions (if $\hat{y}_{i,c} \approx 0$ for the correct class, $\log(\hat{y}_{i,c}) \to -\infty$, giving infinite loss).
- It rewards confident correct predictions (if $\hat{y}_{i,c} \approx 1$, $\log(\hat{y}_{i,c}) \approx 0$, giving near-zero loss).
- It is convex in the model parameters for logistic regression, ensuring a unique global minimum.

**2. Log-Likelihood.** Many models are trained by maximizing the log-likelihood rather than the likelihood. Given i.i.d. data $\{x_i\}_{i=1}^N$ with likelihood $L(\theta) = \prod_i p(x_i | \theta)$, the log-likelihood is:
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^N \log p(x_i | \theta)$$
The logarithm converts the product into a sum, which is much easier to differentiate and optimize. Maximum likelihood estimation (MLE) finds $\theta_{\text{MLE}} = \arg\max_\theta \ell(\theta)$.

**3. Log-Odds (Logit) in Logistic Regression.** Logistic regression models the log-odds (logit) of the probability as a linear function:
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d$$
The logit transformation maps $(0, 1)$ (probability) to $\mathbb{R}$ (log-odds), allowing linear modeling. Solving for $p$ gives:
$$p = \frac{e^{\beta^T x}}{1 + e^{\beta^T x}} = \frac{1}{1 + e^{-\beta^T x}}$$
which is the sigmoid function.

**4. Log-Sum-Exp Trick.** Computing $\log(\sum_i e^{z_i})$ naively can cause numerical overflow. The log-sum-exp trick computes:
$$\log\left(\sum_i e^{z_i}\right) = m + \log\left(\sum_i e^{z_i - m}\right)$$
where $m = \max_i z_i$. This is used in softmax cross-entropy computation, log-partition functions in probabilistic models, and computing marginal likelihoods in latent variable models.

**5. Information-Theoretic Quantities.** Entropy, Kullback-Leibler divergence, and mutual information all use logarithms:
$$H(p) = -\sum_x p(x) \log p(x)$$
$$D_{KL}(p \| q) = \sum_x p(x) \log\frac{p(x)}{q(x)}$$
$$I(X; Y) = \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$$
These quantities are fundamental to decision trees (information gain), variational inference (ELBO), and representation learning.

**6. Feature Transformation.** When features span multiple orders of magnitude (e.g., income, population, frequency), applying a log transformation can make the data more Gaussian-like and improve model performance. This is common in linear regression (log-log models), PCA on skewed data, and natural language processing (term frequency weighting uses $\log(1 + \text{tf})$).

**7. Learning Rate Schedules.** Logarithmic learning rate schedules decay the learning rate as $\eta(t) = \eta_0 / (1 + \alpha \ln(1 + t))$, though exponential schedules are more common.

## Mathematical Explanation

The logarithmic function $f(x) = \log_a(x)$ for $a > 0$, $a \neq 1$ is defined as the inverse of $g(x) = a^x$:

$$f(g(x)) = \log_a(a^x) = x \quad \text{for all } x \in \mathbb{R}$$
$$g(f(x)) = a^{\log_a(x)} = x \quad \text{for all } x > 0$$

**Domain and Range:**
- Domain of $\log_a(x)$: $(0, \infty)$
- Range of $\log_a(x)$: $\mathbb{R}$
- Vertical asymptote at $x = 0$ (the $y$-axis)

**Laws of Logarithms:**

1. **Product Law:** $\log_a(MN) = \log_a(M) + \log_a(N)$
   Proof: Let $u = \log_a M$, $v = \log_a N$. Then $a^u = M$, $a^v = N$. So $MN = a^u a^v = a^{u+v}$. Therefore $\log_a(MN) = u + v = \log_a M + \log_a N$.

2. **Quotient Law:** $\log_a(M/N) = \log_a(M) - \log_a(N)$

3. **Power Law:** $\log_a(M^p) = p \log_a(M)$

4. **Change of Base:** $\log_a(x) = \frac{\log_b(x)}{\log_b(a)}$

5. **Logarithm of 1:** $\log_a(1) = 0$ because $a^0 = 1$.

6. **Logarithm of the Base:** $\log_a(a) = 1$ because $a^1 = a$.

**Natural Logarithm:**
The natural logarithm $\ln x$ has base $e$:
$$\ln x = \log_e x$$

Properties of $\ln x$:
- $\ln(e^x) = x$ for all $x \in \mathbb{R}$
- $e^{\ln x} = x$ for all $x > 0$
- $\frac{d}{dx} \ln x = \frac{1}{x}$ for $x > 0$
- $\int \frac{1}{x} \, dx = \ln |x| + C$
- $\ln x = \int_1^x \frac{1}{t} \, dt$

**Solving Logarithmic Equations:**
- $\log_a x = b \iff x = a^b$ (convert to exponential form).
- If $\log_a f(x) = \log_a g(x)$, then $f(x) = g(x)$ (provided $f, g > 0$).
- Always check for extraneous solutions (arguments of logs must be positive).

**Solving Exponential Equations with Logarithms:**
- $a^x = b \implies x = \log_a b = \frac{\ln b}{\ln a}$.
- $e^{kx} = c \implies kx = \ln c \implies x = \frac{\ln c}{k}$.

## Formula(s)

**General logarithmic function:**
$$f(x) = \log_a(x), \quad a > 0, \, a \neq 1, \, x > 0$$

**Common logarithm:**
$$\log(x) = \log_{10}(x)$$

**Natural logarithm:**
$$\ln(x) = \log_e(x)$$

**Logarithmic identities:**
$$\log_a(MN) = \log_a M + \log_a N$$
$$\log_a\left(\frac{M}{N}\right) = \log_a M - \log_a N$$
$$\log_a(M^p) = p \log_a M$$
$$\log_a(1) = 0, \quad \log_a(a) = 1$$

**Change of base:**
$$\log_a(x) = \frac{\log_b(x)}{\log_b(a)}$$

**Derivative:**
$$\frac{d}{dx} \ln x = \frac{1}{x}, \quad \frac{d}{dx} \log_a x = \frac{1}{x \ln a}$$

**Integral:**
$$\int \frac{1}{x} \, dx = \ln |x| + C$$

**Conversion between exponential and logarithmic forms:**
$$a^y = x \iff \log_a x = y$$

## Properties

1. **Domain:** $(0, \infty)$. The logarithm of a non-positive number is undefined in the real numbers.

2. **Range:** $\mathbb{R}$. The logarithm can produce any real number.

3. **Vertical Asymptote:** $x = 0$ (the $y$-axis). As $x \to 0^+$, $\log_a x \to -\infty$ for $a > 1$, and $\log_a x \to \infty$ for $0 < a < 1$.

4. **$x$-intercept:** $(1, 0)$ for all bases because $\log_a(1) = 0$.

5. **Monotonicity:** Strictly increasing for $a > 1$, strictly decreasing for $0 < a < 1$.

6. **One-to-One:** $\log_a x_1 = \log_a x_2 \implies x_1 = x_2$ (injectivity).

7. **Inverse Relationship:** $f(x) = \log_a x$ and $g(x) = a^x$ are inverse functions. Their graphs are symmetric about the line $y = x$.

8. **Concavity:** For $a > 1$, $\log_a x$ is concave down (second derivative is negative). This means the function "bends" towards the $x$-axis.

9. **Growth Rate:** Logarithmic functions grow slower than any positive power function. For any $a > 1$ and $\varepsilon > 0$, $\lim_{x \to \infty} \frac{\log_a x}{x^\varepsilon} = 0$. This is the slowest-growing class of elementary functions.

10. **Functional Equation:** $\log_a(xy) = \log_a x + \log_a y$. This is the defining property: a logarithm converts multiplication into addition.

## Step-by-Step Worked Examples

### Example 1: Converting Between Exponential and Logarithmic Form

Convert each exponential equation to logarithmic form and vice versa.

**(a) $2^5 = 32$ to logarithmic form:** $\log_2(32) = 5$

**(b) $\log_3(81) = 4$ to exponential form:** $3^4 = 81$

**(c) $e^2 \approx 7.389$ to natural logarithmic form:** $\ln(7.389) \approx 2$

**(d) $\ln(0.5) \approx -0.693$ to exponential form:** $e^{-0.693} \approx 0.5$

### Example 2: Evaluating Logarithms

Evaluate each logarithm without a calculator.

**(a) $\log_2(16)$:** $2^4 = 16$, so $\log_2(16) = 4$.

**(b) $\log_5(1/25)$:** $5^{-2} = 1/25$, so $\log_5(1/25) = -2$.

**(c) $\log_7(1)$:** $7^0 = 1$, so $\log_7(1) = 0$.

**(d) $\log_3(27)$:** $3^3 = 27$, so $\log_3(27) = 3$.

**(e) $\ln(e^5)$:** $\ln(e^5) = 5$ (since $\ln$ and $e^x$ are inverses).

**Answers:** (a) 4, (b) -2, (c) 0, (d) 3, (e) 5.

### Example 3: Applying Logarithm Laws

Simplify $\log_2(8x) + \log_2(x^3) - \log_2(4x^2)$.

**Step 1:** Apply the product and power laws. $\log_2(8x) + \log_2(x^3) = \log_2(8x \cdot x^3) = \log_2(8x^4)$

**Step 2:** Apply the quotient law. $\log_2(8x^4) - \log_2(4x^2) = \log_2\left(\frac{8x^4}{4x^2}\right) = \log_2(2x^2)$

**Answer:** $\log_2(2x^2) = \log_2(2) + \log_2(x^2) = 1 + 2\log_2(x)$

### Example 4: Solving Logarithmic Equations

Solve $\log_2(x) + \log_2(x - 2) = 3$.

**Step 1:** Apply the product law. $\log_2(x(x - 2)) = 3$

**Step 2:** Convert to exponential form. $x(x - 2) = 2^3 = 8$

**Step 3:** Expand and solve. $x^2 - 2x - 8 = 0 \implies (x - 4)(x + 2) = 0 \implies x = 4$ or $x = -2$

**Step 4:** Check for extraneous solutions. Domain requires $x > 0$ and $x - 2 > 0$, so $x > 2$. $x = 4$ is valid. $x = -2$ is extraneous.

**Answer:** $x = 4$

### Example 5: Solving Exponential Equations with Logarithms

Solve $3^{2x} = 7$ for $x$.

**Step 1:** Take the natural logarithm of both sides. $\ln(3^{2x}) = \ln(7)$

**Step 2:** Apply the power law. $2x \ln(3) = \ln(7)$

**Step 3:** Solve for $x$. $x = \frac{\ln(7)}{2 \ln(3)} \approx \frac{1.9459}{2.1972} \approx 0.8857$

**Answer:** $x = \frac{\ln 7}{2 \ln 3} \approx 0.8857$

### Example 6: Change of Base

Evaluate $\log_5(12)$ using common logarithms (base 10).

**Step 1:** Apply the change of base formula. $\log_5(12) = \frac{\log_{10}(12)}{\log_{10}(5)}$

**Step 2:** Compute. $\log_{10}(12) \approx 1.07918$, $\log_{10}(5) \approx 0.69897$. $\frac{1.07918}{0.69897} \approx 1.5440$

**Answer:** $\log_5(12) \approx 1.5440$

### Example 7: Solving a Real-World Exponential Growth Problem

A population of 500 bacteria grows at a rate of 15% per hour. How long until the population reaches 2000?

**Step 1:** Write the growth model. $P(t) = 500 e^{0.15t}$

**Step 2:** Set $P(t) = 2000$ and solve. $500 e^{0.15t} = 2000 \implies e^{0.15t} = 4$

**Step 3:** Take the natural logarithm. $0.15t = \ln(4) \implies t = \frac{\ln(4)}{0.15} \approx \frac{1.3863}{0.15} \approx 9.24$ hours.

**Answer:** Approximately 9.24 hours.

## Visual Interpretation

The graph of $f(x) = \log_a(x)$ for $a > 1$ passes through $(1, 0)$, approaches $-\infty$ as $x \to 0^+$ (vertical asymptote at $x = 0$), increases without bound as $x \to \infty$ (very slowly), and is concave down (bends toward the $x$-axis). For $a > 1$, the graph is the mirror image of $a^x$ across the line $y = x$.

**Comparison of bases ($a > 1$):**
- $\log_2 x$ grows fastest (reaches 1 at $x = 2$).
- $\ln x$ grows slower (reaches 1 at $x = e \approx 2.718$).
- $\log_{10} x$ grows slowest (reaches 1 at $x = 10$).

The natural logarithm $\ln x$ also represents the area under the curve $y = 1/t$ from $t = 1$ to $t = x$. For $x > 1$, this is a positive area. For $0 < x < 1$, the area is negative (integrated backwards).

## Common Mistakes

1. **Assuming $\log_a(x + y) = \log_a x + \log_a y$.** The product law applies to multiplication, not addition: $\log_a(xy) = \log_a x + \log_a y$. There is no simple rule for $\log_a(x + y)$.

2. **Forgetting the domain restriction: arguments of logarithms must be positive.** Many students solve logarithmic equations but forget to check that the solutions make the original arguments positive, leading to extraneous solutions.

3. **Confusing $\ln(x^2)$ with $(\ln x)^2$.** $\ln(x^2) = 2\ln x$ (valid for $x > 0$) is very different from $(\ln x)^2 = (\ln x)(\ln x)$.

4. **Misapplying the change of base formula.** The correct formula is $\log_a(x) = \frac{\log_b(x)}{\log_b(a)}$. A common mistake is inverting the fraction: $\frac{\log_b(a)}{\log_b(x)}$ is wrong.

5. **Thinking $\log_a(0)$ is defined or equals something finite.** $\log_a(0)$ is undefined (it approaches $-\infty$ as $x \to 0^+$).

6. **Assuming the logarithm of a negative number is defined in real numbers.** $\ln(-1)$ is not a real number. In complex analysis, $\ln(-1) = i\pi$, but this is beyond standard real analysis.

7. **Confusing the "log" button on calculators.** Many calculators have $\log$ (base 10) and $\ln$ (base $e$). Using the wrong one without adjusting gives incorrect results.

8. **Incorrectly simplifying $\frac{\log x}{\log y}$.** $\frac{\log x}{\log y} \neq \log(x - y)$ and $\neq \log(x/y)$. The correct identity is $\log_y x = \frac{\log x}{\log y}$.

9. **Forgetting that $\log_a(x)$ is undefined when $a = 1$ or $a \leq 0$.** Just like exponential functions, logarithms require $a > 0$, $a \neq 1$.

10. **Numerical issues with cross-entropy loss.** When the predicted probability $\hat{y}$ is exactly 0, $\log(\hat{y}) = -\infty$, causing numerical instability. ML frameworks clip predictions to a small positive value (e.g., $10^{-7}$) to avoid this.

## Interview Questions

### Beginner

1. **What is a logarithm? Give the definition.**
   *Answer: $\log_a(x)$ is the exponent to which the base $a$ must be raised to obtain $x$. Equivalently, $y = \log_a(x)$ means $a^y = x$. The domain is $x > 0$, and the range is all real numbers.*

2. **Evaluate $\log_3(9)$ and $\log_3(1/27)$.**
   *Answer: $\log_3(9) = 2$ because $3^2 = 9$. $\log_3(1/27) = -3$ because $3^{-3} = 1/27$.*

3. **What is the natural logarithm? How is it related to $e$?**
   *Answer: The natural logarithm $\ln x$ is the logarithm with base $e$ (Euler's number, $\approx 2.718$). It is the inverse of $e^x$: $\ln(e^x) = x$ and $e^{\ln x} = x$.*

4. **Simplify $\log_2(8) + \log_2(4)$.**
   *Answer: $\log_2(8) = 3$ and $\log_2(4) = 2$, so the sum is $5$. Alternatively, $\log_2(8 \cdot 4) = \log_2(32) = 5$.*

5. **What is the domain of $f(x) = \ln(x - 5)$?**
   *Answer: The argument must be positive: $x - 5 > 0 \implies x > 5$. Domain: $(5, \infty)$.*

### Intermediate

1. **Explain the cross-entropy loss function and why it uses logarithms.**
   *Answer: Cross-entropy loss is $\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$. The logarithm heavily penalizes confident wrong predictions, rewards confident correct predictions, and makes the loss convex for logistic regression.*

2. **What is the log-sum-exp trick and when is it used?**
   *Answer: The log-sum-exp trick computes $\log(\sum_i e^{z_i})$ stably as $m + \log(\sum_i e^{z_i - m})$ where $m = \max_i z_i$. It is used in softmax cross-entropy and log-partition functions.*

3. **Solve $\log_2(x + 1) - \log_2(x - 1) = 2$.**
   *Answer: $\log_2\left(\frac{x+1}{x-1}\right) = 2 \implies \frac{x+1}{x-1} = 4 \implies x = 5/3$. Check: $5/3 + 1 = 8/3 > 0$, $5/3 - 1 = 2/3 > 0$. Answer: $x = 5/3$.*

4. **How does the logit function map probabilities to real numbers, and why is this useful in logistic regression?**
   *Answer: The logit is $\text{logit}(p) = \ln(p/(1-p))$, mapping $(0,1)$ to $\mathbb{R}$. In logistic regression, modeling the log-odds linearly allows using linear regression techniques for binary classification.*

5. **Derive the formula for the derivative of $\ln x$ from the definition of the logarithm as an integral.**
   *Answer: By definition, $\ln x = \int_1^x \frac{1}{t} dt$. By the Fundamental Theorem of Calculus, $\frac{d}{dx} \ln x = \frac{1}{x}$ for $x > 0$.*

### Advanced

1. **Explain the relationship between maximum likelihood estimation (MLE) and the logarithm. Why do we maximize the log-likelihood instead of the likelihood?**
   *Answer: For i.i.d. data, the likelihood is $L(\theta) = \prod_i p(x_i | \theta)$. The log converts products to sums: $\ell(\theta) = \sum_i \log p(x_i | \theta)$. We maximize $\ell(\theta)$ because sums are easier to differentiate, the log-likelihood is numerically stable, and it is concave for exponential family distributions.*

2. **Prove the product law for logarithms using the definition of a logarithm as an inverse of the exponential function.**
   *Proof: Let $u = \log_a M$, $v = \log_a N$. Then $a^u = M$, $a^v = N$. So $MN = a^u a^v = a^{u+v}$. Taking $\log_a$: $\log_a(MN) = u + v = \log_a M + \log_a N$.*

3. **How does the logarithm help with numerical stability in gradient-based optimization? Give a concrete example involving the softmax function.**
   *Answer: The combined log-softmax $\log(\sigma(z)_i) = z_i - \log(\sum_j e^{z_j})$ uses the log-sum-exp trick to avoid overflow from large exponentials. The gradient $\hat{y} - y$ is numerically stable. PyTorch and TensorFlow both use this fused computation.*

## Practice Problems

### Easy

1. Evaluate $\log_4(64)$, $\log_{10}(0.001)$, and $\ln(e^3)$.
2. Convert $5^3 = 125$ to logarithmic form.
3. Find the domain of $f(x) = \log_3(2x + 6)$.
4. Simplify $\ln(ab) + \ln(a^2) - \ln(b)$.
5. Solve $\log_2(x) = 5$.

### Medium

1. Solve $\log_3(x) + \log_3(x - 6) = 3$.
2. Solve $4 \cdot 2^{3x} = 16$ using logarithms.
3. Evaluate $\log_4(9)$ using base 2 (change of base).
4. Express $\ln\left(\frac{x^2 \sqrt{y}}{z^3}\right)$ in terms of $\ln x$, $\ln y$, and $\ln z$.
5. A radioactive sample decays according to $N(t) = N_0 e^{-0.02t}$. How long until only 25% remains?

### Hard

1. Solve for $x$: $\log_2(x) + \log_4(x) + \log_8(x) = 11$.
2. Prove that $\frac{d}{dx} \log_a x = \frac{1}{x \ln a}$ using the change of base formula.
3. Derive the gradient with respect to logits $z_i$ for cross-entropy loss with sigmoid output: $\hat{y}_i = \sigma(z_i)$, $\mathcal{L}_i = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$.

## Solutions

### Easy Solutions

**1.** $\log_4(64) = 3$, $\log_{10}(0.001) = -3$, $\ln(e^3) = 3$.

**2.** $\log_5(125) = 3$.

**3.** $2x + 6 > 0 \implies x > -3$. Domain: $(-3, \infty)$.

**4.** $\ln(ab) + \ln(a^2) - \ln(b) = \ln(a^3) = 3 \ln a$.

**5.** $\log_2(x) = 5 \implies x = 2^5 = 32$.

### Medium Solutions

**1.** $\log_3(x(x - 6)) = 3 \implies x(x-6) = 27 \implies x^2 - 6x - 27 = 0 \implies (x-9)(x+3) = 0$. $x = 9$ (valid), $x = -3$ (extraneous). Answer: $x = 9$.

**2.** $4 \cdot 2^{3x} = 16 \implies 2^{3x} = 4 = 2^2 \implies 3x = 2 \implies x = 2/3$.

**3.** $\log_4(9) = \frac{\log_2(9)}{\log_2(4)} = \frac{\log_2(9)}{2} = \log_2(3) \approx 1.585$.

**4.** $\ln\left(\frac{x^2 \sqrt{y}}{z^3}\right) = 2\ln x + \frac{1}{2}\ln y - 3\ln z$.

**5.** $0.25 N_0 = N_0 e^{-0.02t} \implies e^{-0.02t} = 0.25 \implies -0.02t = \ln(0.25) \implies t = 50 \ln 4 \approx 69.3$ time units.

### Hard Solutions

**1.** Convert to base 2: $\log_4 x = \frac{\log_2 x}{2}$, $\log_8 x = \frac{\log_2 x}{3}$. Then $\log_2 x (1 + 1/2 + 1/3) = 11 \implies \log_2 x \cdot \frac{11}{6} = 11 \implies \log_2 x = 6 \implies x = 64$.

**2.** $\log_a x = \frac{\ln x}{\ln a}$. Then $\frac{d}{dx} \log_a x = \frac{1}{\ln a} \cdot \frac{d}{dx} \ln x = \frac{1}{\ln a} \cdot \frac{1}{x} = \frac{1}{x \ln a}$.

**3.** $\frac{d\mathcal{L}_i}{dz} = \frac{d\mathcal{L}_i}{d\hat{y}} \cdot \frac{d\hat{y}}{dz} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y}) = \hat{y} - y$. This elegant result shows the gradient is simply the prediction error.

## Related Concepts

- **Exponential Function** (MATH-050) — The inverse of the logarithmic function; $a^x$ and $\log_a x$ are inverses.
- **Function** (MATH-044) — Logarithms are a special class of functions with specific domain and range.
- **Inverse Function** (MATH-048) — The logarithmic function is the inverse of the exponential function.
- **Cross-Entropy Loss** — The standard loss function for classification, using the logarithm to measure prediction error.
- **Logistic Regression** — Uses the log-odds (logit) transformation to model binary outcomes.
- **Information Theory** — Entropy and KL divergence are defined using logarithms.

## Next Concepts

- **Trigonometric Function** (MATH-052) — Periodic functions with logarithmic relationships through complex exponentials.
- **Hyperbolic Functions** — Defined using exponentials: $\sinh x = (e^x - e^{-x})/2$, $\cosh x = (e^x + e^{-x})/2$.
- **Logistic Function** — The sigmoid $\sigma(x) = 1/(1+e^{-x})$ combines exponentials and logarithms.
- **Complex Logarithms** — Extending logarithms to the complex plane via Euler's formula.

## Summary

The logarithmic function $f(x) = \log_a(x)$ is the inverse of the exponential function $a^x$, with domain $(0, \infty)$ and range $\mathbb{R}$. It converts multiplication into addition, division into subtraction, and exponentiation into multiplication. The natural logarithm $\ln x$ (base $e$) is the most important in mathematics and ML because its derivative is $1/x$ and it provides the link between exponentials and probability. In machine learning, logarithms are essential for cross-entropy loss, log-likelihood maximization, the logit function in logistic regression, the log-sum-exp trick for numerical stability, and information-theoretic quantities like entropy and KL divergence.

## Key Takeaways

- $\log_a(x)$ is the inverse of $a^x$: $a^{\log_a x} = x$, $\log_a(a^x) = x$.
- Domain: $(0, \infty)$, Range: $\mathbb{R}$.
- Key identities: $\log(MN) = \log M + \log N$, $\log(M^p) = p \log M$.
- Change of base: $\log_a x = \frac{\log_b x}{\log_b a}$.
- $\ln x = \log_e x$ is the natural logarithm; $\frac{d}{dx} \ln x = \frac{1}{x}$.
- Cross-entropy loss uses $\log$ to penalize wrong confident predictions.
- Log-sum-exp trick ensures numerical stability in softmax computations.
- Log-odds (logit) transforms probabilities to real numbers for logistic regression.
- Logarithms grow slower than any positive power function.
- Always check domain: arguments of logs must be strictly positive.
