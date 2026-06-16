# Concept: Integral Calculus

## Concept ID

MATH-061

## Domain

Mathematics

## Module

Calculus

## Difficulty

INTERMEDIATE

## Learning Objectives

- Understand the integral as the antiderivative and as the area under a curve
- Distinguish between definite and indefinite integrals
- Apply basic integration rules including power rule, exponential rule, and logarithmic rule
- Use substitution and integration by parts to solve integrals
- Recognise the relationship between integration and differentiation
- Apply integration in probability and machine learning contexts

## Prerequisites

- Differential calculus (derivatives, chain rule, product rule)
- Basic algebra and function analysis
- Understanding of limits and continuity
- Familiarity with elementary functions (polynomials, exponentials, logarithms, trigonometric functions)

## Definition

An **integral** is a fundamental concept in calculus that generalises the notions of area, accumulation, and antiderivatives. There are two complementary types:

The **indefinite integral** (antiderivative) of a function $f(x)$ with respect to $x$ is a function $F(x)$ such that $F'(x) = f(x)$, written as:

$$
\int f(x) \, dx = F(x) + C
$$

where $C$ is the constant of integration.

The **definite integral** of $f(x)$ from $a$ to $b$ represents the signed area under the curve of $f$ between $x = a$ and $x = b$:

$$
\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x
$$

where the limit refines a Riemann sum approximation.

The two concepts are unified by the **Fundamental Theorem of Calculus**, which states that differentiation and integration are inverse operations.

## Intuition

Integration answers two intimately related questions: "What function has this derivative?" and "What is the accumulated area under this curve?"

Think of integration as accumulation. If $f(t)$ represents the rate of change of some quantity at time $t$ (e.g., velocity), then the integral $\int_a^b f(t) \, dt$ gives the total change in that quantity from $t=a$ to $t=b$ (e.g., displacement). This is why integrating velocity gives position.

Alternatively, imagine summing the areas of infinitely thin rectangles under a curve $y = f(x)$. As the rectangles become narrower, the approximation improves, and in the limit we get the exact area. The integral captures this limiting process precisely.

The constant of integration $C$ reflects the fact that antiderivatives are not unique — adding any constant to $F(x)$ yields another function with the same derivative, since the derivative of a constant is zero.

## Why This Concept Matters

Integration is one of the two pillars of calculus (alongside differentiation) and is indispensable across all quantitative fields. In physics, integrals compute work, energy, charge, and centre of mass. In engineering, they are used for signal processing, control theory, and structural analysis. In economics, they calculate consumer surplus, present value, and growth over time.

For data science and machine learning, integration is the mathematical foundation of probability theory. Every probability density function must integrate to 1 (normalisation). Expected values, variances, and moments are all defined as integrals. The marginalisation of joint distributions — a core operation in Bayesian inference — is integration. Variational inference, a key technique for approximating intractable distributions, involves optimising the Evidence Lower Bound (ELBO), which itself is defined through an integral.

Understanding integration is essential for reading and contributing to the machine learning literature, where integrals appear in loss functions, regularisation terms, and theoretical analyses.

## Historical Background

Integral calculus originated in ancient Greece with the method of exhaustion, pioneered by **Eudoxus** (c. 390–337 BCE) and extensively used by **Archimedes** (c. 287–212 BCE) to compute areas and volumes of geometric figures. Archimedes famously computed the area of a parabolic segment using what we would now recognise as a limiting process.

The modern development of integral calculus began in the 17th century with **Isaac Newton** (1642–1727) and **Gottfried Wilhelm Leibniz** (1646–1716), who independently discovered the Fundamental Theorem of Calculus, linking integration and differentiation. Leibniz introduced the integral symbol $\int$ (an elongated S, from the Latin _summa_, meaning sum).

The rigorous foundations of integration were established in the 19th century by **Augustin-Louis Cauchy** (1789–1857), who defined the definite integral using limits of sums, and **Bernhard Riemann** (1826–1866), who formalised the Riemann integral, the standard definition used today. Later, **Henri Lebesgue** (1875–1941) developed a more powerful theory of integration (Lebesgue integration) that generalises the Riemann integral and is essential for modern probability theory and functional analysis.

## Real World Examples

1. **Physics: Computing Distance from Velocity** — If a car's velocity is $v(t) = 3t^2$ m/s, the distance travelled from $t = 0$ to $t = 5$ seconds is $\int_0^5 3t^2 \, dt = [t^3]_0^5 = 125$ metres.

2. **Economics: Consumer Surplus** — Consumer surplus is the integral of the demand curve minus the market price over the quantity sold. It measures the benefit consumers receive beyond what they pay.

3. **Biology: Drug Concentration** — The total exposure to a drug (area under the concentration-time curve, or AUC) is the integral of the drug concentration over time. This integral is critical for determining dosage protocols.

4. **Environmental Science: Carbon Emissions** — The total carbon emitted over a time period is the integral of the emission rate over that period. Climate models routinely compute such integrals.

5. **Traffic Engineering: Queue Length** — The number of cars waiting at a traffic light is the integral of the arrival rate minus the departure rate over time. Traffic engineers use this to optimise signal timing.

## AI/ML Relevance

1. **Probability Normalisation** — Every probability density function must satisfy $\int_{-\infty}^{\infty} p(x) \, dx = 1$. Computing normalisation constants often requires integration, especially in Bayesian inference where $p(\theta|x) = \frac{p(x|\theta)p(\theta)}{\int p(x|\theta)p(\theta) \, d\theta}$.

2. **Expected Values** — The expected value of a continuous random variable is $\mathbb{E}[X] = \int_{-\infty}^{\infty} x p(x) \, dx$. Expected values appear in loss functions (e.g., mean squared error), risk minimization, and reinforcement learning (value functions).

3. **Variational Inference and the ELBO** — The Evidence Lower Bound is defined as:
   $$
   \text{ELBO}(q) = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z})] = \int q(\mathbf{z}) \log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})} \, d\mathbf{z}
   $$
   Optimising the ELBO involves computing expectations (integrals) over the variational distribution.

4. **Entropy and Information Theory** — The differential entropy of a continuous distribution is $H(p) = -\int p(x) \log p(x) \, dx$, and the Kullback-Leibler divergence is $D_{KL}(p\|q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$.

5. **Gaussian Integrals** — The Gaussian integral $\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$ is the foundation of the normal distribution. More generally, integrals of quadratic forms in exponentials arise throughout machine learning, from kernel methods to Gaussian processes.

6. **Kernel Methods** — The kernel trick can be viewed as computing an integral over a feature space: $k(x, y) = \int \phi(x)^T \phi(y) \, d\mu$ for certain feature maps.

## Mathematical Explanation

### The Indefinite Integral

The indefinite integral (or antiderivative) reverses differentiation. If $F'(x) = f(x)$, then $\int f(x) \, dx = F(x) + C$. The constant $C$ accounts for all possible antiderivatives.

### Basic Integration Rules

1. **Power Rule** (for $n \neq -1$):
   $$
   \int x^n \, dx = \frac{x^{n+1}}{n+1} + C
   $$

2. **Logarithmic Rule** (for $n = -1$):
   $$
   \int \frac{1}{x} \, dx = \ln|x| + C
   $$

3. **Exponential Rule**:
   $$
   \int e^x \, dx = e^x + C, \quad \int a^x \, dx = \frac{a^x}{\ln a} + C
   $$

4. **Constant Multiple Rule**:
   $$
   \int k f(x) \, dx = k \int f(x) \, dx
   $$

5. **Sum/Difference Rule**:
   $$
   \int [f(x) \pm g(x)] \, dx = \int f(x) \, dx \pm \int g(x) \, dx
   $$

### Integration by Substitution

Also called the reverse chain rule. If $u = g(x)$ and $du = g'(x) \, dx$:

$$
\int f(g(x)) g'(x) \, dx = \int f(u) \, du
$$

### Integration by Parts

The reverse product rule. From $(uv)' = u'v + uv'$:

$$
\int u \, dv = uv - \int v \, du
$$

### Trigonometric Integrals

Standard results:
- $\int \sin x \, dx = -\cos x + C$
- $\int \cos x \, dx = \sin x + C$
- $\int \sec^2 x \, dx = \tan x + C$
- $\int \csc^2 x \, dx = -\cot x + C$
- $\int \sec x \tan x \, dx = \sec x + C$
- $\int \csc x \cot x \, dx = -\csc x + C$

### Partial Fractions

For rational functions $\frac{P(x)}{Q(x)}$ where $\deg(P) < \deg(Q)$, factor $Q(x)$ and decompose into simpler fractions, then integrate term by term.

## Formula(s)

1. **Indefinite Integral Definition**:
   $$
   \int f(x) \, dx = F(x) + C \iff F'(x) = f(x)
   $$

2. **Power Rule**:
   $$
   \int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)
   $$

3. **Logarithmic Integration**:
   $$
   \int \frac{1}{x} \, dx = \ln|x| + C
   $$

4. **Exponential Integration**:
   $$
   \int e^{ax} \, dx = \frac{1}{a} e^{ax} + C
   $$

5. **Substitution Rule**:
   $$
   \int f(g(x)) g'(x) \, dx = \int f(u) \, du, \quad u = g(x)
   $$

6. **Integration by Parts**:
   $$
   \int u \, dv = uv - \int v \, du
   $$

7. **Linearity of Integration**:
   $$
   \int [af(x) + bg(x)] \, dx = a\int f(x) \, dx + b\int g(x) \, dx
   $$

## Properties

1. **Linearity**: The integral of a sum is the sum of integrals; constants factor out.

2. **Antiderivative Relation**: $\frac{d}{dx} \int f(x) \, dx = f(x)$ (differentiation undoes indefinite integration).

3. **Constant of Integration**: Indefinite integrals are unique only up to an additive constant.

4. **Domain Considerations**: The indefinite integral is defined on intervals where $f$ is continuous.

5. **Integration Preserves Parity**: If $f$ is even, its antiderivative (with $C=0$) is odd; if $f$ is odd, its antiderivative is even.

6. **Reversal of Differentiation**: $\int f'(x) \, dx = f(x) + C$ (integration undoes differentiation up to a constant).

7. **Non-Uniqueness**: A function can have infinitely many antiderivatives, all differing by a constant.

## Step-by-Step Worked Examples

### Example 1: Basic Power Rule

**Problem**: Compute $\int (3x^4 - 2x^2 + 5) \, dx$.

**Solution**:

Step 1: Apply linearity to split the integral.

$$
\int (3x^4 - 2x^2 + 5) \, dx = 3\int x^4 \, dx - 2\int x^2 \, dx + 5\int 1 \, dx
$$

Step 2: Apply the power rule to each term.

For $n = 4$: $\int x^4 \, dx = \frac{x^{5}}{5} + C_1$
For $n = 2$: $\int x^2 \, dx = \frac{x^{3}}{3} + C_2$
For $n = 0$: $\int 1 \, dx = \int x^0 \, dx = x + C_3$

Step 3: Combine with constants.

$$
3 \cdot \frac{x^{5}}{5} - 2 \cdot \frac{x^{3}}{3} + 5x + C
$$

where $C = 3C_1 - 2C_2 + 5C_3$ is a combined constant.

Result:
$$
\int (3x^4 - 2x^2 + 5) \, dx = \frac{3x^5}{5} - \frac{2x^3}{3} + 5x + C
$$

### Example 2: Integration by Substitution

**Problem**: Compute $\int 2x e^{x^2} \, dx$.

**Solution**:

Step 1: Identify a suitable substitution. Notice that $2x$ is the derivative of $x^2$. Let $u = x^2$.

Step 2: Compute $du$. $du = 2x \, dx$.

Step 3: Rewrite the integral in terms of $u$.

$$
\int 2x e^{x^2} \, dx = \int e^{u} \, du
$$

Step 4: Integrate with respect to $u$.

$$
\int e^{u} \, du = e^{u} + C
$$

Step 5: Substitute back $u = x^2$.

$$
\int 2x e^{x^2} \, dx = e^{x^2} + C
$$

Verification: $\frac{d}{dx} e^{x^2} = e^{x^2} \cdot 2x = 2x e^{x^2}$, which matches the original integrand.

### Example 3: Integration by Parts

**Problem**: Compute $\int x \ln x \, dx$.

**Solution**:

Step 1: Choose $u$ and $dv$ for integration by parts. Let $u = \ln x$ and $dv = x \, dx$.

Step 2: Compute $du$ and $v$.
$du = \frac{1}{x} \, dx$, $v = \int x \, dx = \frac{x^2}{2}$.

Step 3: Apply integration by parts formula $\int u \, dv = uv - \int v \, du$.

$$
\int x \ln x \, dx = \ln x \cdot \frac{x^2}{2} - \int \frac{x^2}{2} \cdot \frac{1}{x} \, dx
$$

Step 4: Simplify the remaining integral.

$$
= \frac{x^2 \ln x}{2} - \frac{1}{2} \int x \, dx
$$

$$
= \frac{x^2 \ln x}{2} - \frac{1}{2} \cdot \frac{x^2}{2} + C
$$

Step 5: Final result.

$$
\int x \ln x \, dx = \frac{x^2 \ln x}{2} - \frac{x^2}{4} + C = \frac{x^2}{4} (2 \ln x - 1) + C
$$

### Example 4: Substitution with Trigonometric Functions

**Problem**: Compute $\int \sin^3 x \cos x \, dx$.

**Solution**:

Step 1: Let $u = \sin x$. Then $du = \cos x \, dx$.

Step 2: Rewrite.

$$
\int \sin^3 x \cos x \, dx = \int u^3 \, du
$$

Step 3: Integrate.

$$
\int u^3 \, du = \frac{u^4}{4} + C
$$

Step 4: Substitute back.

$$
\int \sin^3 x \cos x \, dx = \frac{\sin^4 x}{4} + C
$$

### Example 5: Integration Using Partial Fractions

**Problem**: Compute $\int \frac{5x + 3}{x^2 + 2x - 3} \, dx$.

**Solution**:

Step 1: Factor the denominator.

$x^2 + 2x - 3 = (x+3)(x-1)$

Step 2: Decompose into partial fractions.

$$
\frac{5x + 3}{(x+3)(x-1)} = \frac{A}{x+3} + \frac{B}{x-1}
$$

Multiply both sides by $(x+3)(x-1)$:

$$
5x + 3 = A(x-1) + B(x+3) = Ax - A + Bx + 3B = (A+B)x + (-A + 3B)
$$

Step 3: Solve for $A$ and $B$ by equating coefficients.

Coefficient of $x$: $A + B = 5$
Constant term: $-A + 3B = 3$

Adding: $4B = 8 \implies B = 2$, then $A = 3$.

Step 4: Integrate.

$$
\int \frac{5x + 3}{x^2 + 2x - 3} \, dx = \int \frac{3}{x+3} \, dx + \int \frac{2}{x-1} \, dx
$$

$$
= 3 \ln|x+3| + 2 \ln|x-1| + C
$$

## Visual Interpretation

The indefinite integral can be visualised as a family of curves, all having the same slope at each $x$ value. If $f(x)$ is the derivative (slope) function, then $\int f(x) \, dx = F(x) + C$ represents all vertical shifts of the antiderivative $F(x)$. At any $x$, each curve $F(x) + C$ has the same tangent slope $f(x)$, but they are separated vertically by different constants.

Geometrically, the integral measures the area under the curve $y = f(x)$. For a small change $dx$, the contribution to the total area is $f(x) \, dx$, the area of a thin rectangle. Summing all these infinitesimal rectangles from some starting point gives the accumulated area.

You can think of integration as a machine: you feed it a rate function $f(x)$, and it produces an accumulation function $F(x) + C$ that tells you the total accumulated quantity from a reference point up to $x$.

## Common Mistakes

1. **Forgetting the constant of integration $C$**: Every indefinite integral must include $+C$. Omitting it is a common error because antiderivatives are not unique.

2. **Applying the power rule to $n = -1$**: $\int x^{-1} \, dx = \ln|x| + C$, not $\frac{x^0}{0} + C$, which is undefined. This is the single exception to the power rule.

3. **Incorrect substitution**: Forgetting to fully express the integral in terms of the new variable, including $du$. A common mistake is substituting only part of the integrand while leaving $dx$ unchanged.

4. **Misapplying integration by parts**: Choosing $u$ and $dv$ poorly can make the integral harder rather than easier. The LIATE rule (Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential) helps prioritise $u$ choices.

5. **Algebraic errors in partial fractions**: Failing to ensure the numerator degree is less than the denominator degree (requiring polynomial division first), or making mistakes solving for the coefficients.

6. **Confusing the variable of integration**: When computing definite integrals, substituting the limits correctly is crucial. For indefinite integrals, forgetting to substitute back from $u$ to $x$ is a common oversight.

7. **Sign errors in integration by parts**: The formula is $\int u \, dv = uv - \int v \, du$. The minus sign before the second integral is frequently mishandled.

8. **Assuming all functions are integrable in closed form**: Many functions (e.g., $e^{-x^2}$, $\frac{\sin x}{x}$) do not have elementary antiderivatives. Numerical or special function methods are required.

## Interview Questions

### Beginner

1. **Q**: What is the difference between definite and indefinite integration?
   **A**: Indefinite integration finds a family of antiderivatives $\int f(x) \, dx = F(x) + C$, while definite integration computes a signed area $\int_a^b f(x) \, dx$, which is a single number.

2. **Q**: Compute $\int (2x^3 - 4x + 7) \, dx$.
   **A**: $\frac{2x^4}{4} - \frac{4x^2}{2} + 7x + C = \frac{x^4}{2} - 2x^2 + 7x + C$.

3. **Q**: What is $\int e^{2x} \, dx$?
   **A**: $\frac{1}{2} e^{2x} + C$, using the rule $\int e^{ax} \, dx = \frac{1}{a} e^{ax} + C$.

4. **Q**: Why do we include $+C$ in indefinite integrals?
   **A**: Because if $F'(x) = f(x)$, then $(F(x) + C)' = F'(x) = f(x)$ for any constant $C$. The derivative of any constant is zero, so antiderivatives are only unique up to an additive constant.

5. **Q**: What is the integral of $\frac{1}{x}$ and why does it exclude $x = 0$?
   **A**: $\int \frac{1}{x} \, dx = \ln|x| + C$. The absolute value is needed because $\ln x$ is only defined for $x > 0$, but $\frac{1}{x}$ is defined for $x \neq 0$. The integral is valid on intervals not containing $0$.

### Intermediate

1. **Q**: Use substitution to compute $\int \frac{2x}{1 + x^2} \, dx$.
   **A**: Let $u = 1 + x^2$, $du = 2x \, dx$. Then $\int \frac{du}{u} = \ln|u| + C = \ln(1 + x^2) + C$.

2. **Q**: Compute $\int x e^x \, dx$ using integration by parts.
   **A**: Let $u = x$, $dv = e^x \, dx$, so $du = dx$, $v = e^x$. Then $\int x e^x \, dx = x e^x - \int e^x \, dx = x e^x - e^x + C = e^x(x - 1) + C$.

3. **Q**: Why is integration important for probability and machine learning?
   **A**: Probability densities must integrate to 1 (normalisation). Expected values, variances, entropy, and KL divergence are all defined as integrals. Variational inference optimises the ELBO, an integral over hidden variables.

4. **Q**: Compute $\int \tan x \, dx$.
   **A**: $\tan x = \frac{\sin x}{\cos x}$. Let $u = \cos x$, $du = -\sin x \, dx$. Then $\int \tan x \, dx = -\int \frac{du}{u} = -\ln|\cos x| + C = \ln|\sec x| + C$.

5. **Q**: What is the difference between Riemann and Lebesgue integration?
   **A**: The Riemann integral partitions the domain (x-axis) into intervals and sums rectangles. The Lebesgue integral partitions the range (y-axis), grouping points with similar function values. Lebesgue integration handles a wider class of functions and is the foundation of modern probability theory.

### Advanced

1. **Q**: Derive the formula for integration by parts from the product rule.
   **A**: The product rule: $(u(x)v(x))' = u'(x)v(x) + u(x)v'(x)$. Integrate both sides: $u(x)v(x) = \int v(x) u'(x) \, dx + \int u(x) v'(x) \, dx$. Rearranging with $du = u'(x)dx$, $dv = v'(x)dx$: $\int u \, dv = uv - \int v \, du$.

2. **Q**: In variational inference, the ELBO is $\mathbb{E}_{q}[ \log p(x, z) - \log q(z) ]$. Why can this integral be difficult to compute, and what approximation techniques are used?
   **A**: The integral is over the latent variables $z$, which may be high-dimensional. Exact integration is intractable for most models. Techniques include: mean-field variational family (factorised $q$ that makes the integral separable), reparameterisation trick (turning the expectation into an integral over a standard distribution), Monte Carlo estimation, and conjugate exponential family models where closed-form updates exist.

3. **Q**: The Gaussian integral $\int_{-\infty}^{\infty} e^{-x^2/2} \, dx = \sqrt{2\pi}$ is ubiquitous in ML. Compute this using a polar coordinate trick and explain its relevance to Bayesian inference.
   **A**: Let $I = \int_{-\infty}^{\infty} e^{-x^2/2} \, dx$. Then $I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2+y^2)/2} \, dx \, dy$. Convert to polar: $x = r\cos\theta$, $y = r\sin\theta$, $dx\,dy = r\,dr\,d\theta$. Then $I^2 = \int_0^{2\pi} \int_0^{\infty} re^{-r^2/2} \, dr \, d\theta = 2\pi [-e^{-r^2/2}]_0^{\infty} = 2\pi$. So $I = \sqrt{2\pi}$. This normalises the standard normal distribution. In Bayesian inference, the normalisation constant $p(x) = \int p(x|\theta)p(\theta) \, d\theta$ (marginal likelihood) often involves Gaussian integrals and determines model evidence for model comparison.

## Practice Problems

### Easy

1. Compute $\int (5x^3 - 3x + 2) \, dx$.

2. Compute $\int (e^x + \frac{2}{x}) \, dx$.

3. Compute $\int \cos(3x) \, dx$.

4. Compute $\int \frac{4}{x^2} \, dx$ (rewrite as $4x^{-2}$).

5. Compute $\int \sqrt{x} \, dx$.

### Medium

1. Use substitution to compute $\int x \cos(x^2 + 1) \, dx$.

2. Use integration by parts to compute $\int x^2 \ln x \, dx$.

3. Compute $\int \frac{3x + 2}{(x+1)(x+2)} \, dx$ using partial fractions.

4. Compute $\int \frac{\ln x}{x} \, dx$.

5. Compute $\int x e^{3x} \, dx$.

### Hard

1. Use repeated integration by parts to compute $\int x^2 e^{2x} \, dx$.

2. Compute $\int \frac{dx}{x^2 + 4x + 5}$ (complete the square).

3. Compute $\int e^x \sin x \, dx$ (requires integration by parts twice and solving for the original integral).

## Solutions

### Easy Solutions

**Solution 1**: $\int (5x^3 - 3x + 2) \, dx = \frac{5x^4}{4} - \frac{3x^2}{2} + 2x + C$.

**Solution 2**: $\int (e^x + \frac{2}{x}) \, dx = e^x + 2\ln|x| + C$.

**Solution 3**: Let $u = 3x$, $du = 3\, dx$, $dx = \frac{du}{3}$. $\int \cos(3x) \, dx = \frac{1}{3} \int \cos u \, du = \frac{1}{3} \sin u + C = \frac{1}{3} \sin(3x) + C$.

**Solution 4**: $\int \frac{4}{x^2} \, dx = 4\int x^{-2} \, dx = 4 \cdot \frac{x^{-1}}{-1} + C = -\frac{4}{x} + C$.

**Solution 5**: $\int \sqrt{x} \, dx = \int x^{1/2} \, dx = \frac{x^{3/2}}{3/2} + C = \frac{2}{3} x^{3/2} + C$.

### Medium Solutions

**Solution 1**: Let $u = x^2 + 1$, $du = 2x \, dx$, so $x \, dx = \frac{du}{2}$.
$\int x \cos(x^2 + 1) \, dx = \frac{1}{2} \int \cos u \, du = \frac{1}{2} \sin u + C = \frac{1}{2} \sin(x^2 + 1) + C$.

**Solution 2**: Let $u = \ln x$, $dv = x^2 \, dx$, so $du = \frac{1}{x} \, dx$, $v = \frac{x^3}{3}$.
$\int x^2 \ln x \, dx = \frac{x^3}{3} \ln x - \int \frac{x^3}{3} \cdot \frac{1}{x} \, dx = \frac{x^3}{3} \ln x - \frac{1}{3} \int x^2 \, dx = \frac{x^3}{3} \ln x - \frac{x^3}{9} + C = \frac{x^3}{9}(3\ln x - 1) + C$.

**Solution 3**: Decompose: $\frac{3x+2}{(x+1)(x+2)} = \frac{A}{x+1} + \frac{B}{x+2}$, giving $A=1$, $B=2$.
Then $\int \frac{3x+2}{(x+1)(x+2)} \, dx = \int \frac{1}{x+1} \, dx + \int \frac{2}{x+2} \, dx = \ln|x+1| + 2\ln|x+2| + C$.

**Solution 4**: Let $u = \ln x$, $du = \frac{1}{x} \, dx$.
$\int \frac{\ln x}{x} \, dx = \int u \, du = \frac{u^2}{2} + C = \frac{(\ln x)^2}{2} + C$.

**Solution 5**: Let $u = x$, $dv = e^{3x} \, dx$, so $du = dx$, $v = \frac{1}{3}e^{3x}$.
$\int x e^{3x} \, dx = \frac{x}{3}e^{3x} - \int \frac{1}{3}e^{3x} \, dx = \frac{x}{3}e^{3x} - \frac{1}{9}e^{3x} + C = \frac{e^{3x}}{9}(3x - 1) + C$.

### Hard Solutions

**Solution 1**: Let $u = x^2$, $dv = e^{2x} \, dx$, $du = 2x \, dx$, $v = \frac{1}{2}e^{2x}$.
$\int x^2 e^{2x} \, dx = \frac{x^2}{2}e^{2x} - \int x e^{2x} \, dx$.
Now compute $\int x e^{2x} \, dx$: let $u = x$, $dv = e^{2x} \, dx$, $du = dx$, $v = \frac{1}{2}e^{2x}$.
$\int x e^{2x} \, dx = \frac{x}{2}e^{2x} - \int \frac{1}{2}e^{2x} \, dx = \frac{x}{2}e^{2x} - \frac{1}{4}e^{2x} + C_1$.
Substitute back:
$\int x^2 e^{2x} \, dx = \frac{x^2}{2}e^{2x} - \frac{x}{2}e^{2x} + \frac{1}{4}e^{2x} + C = \frac{e^{2x}}{4}(2x^2 - 2x + 1) + C$.

**Solution 2**: Complete the square: $x^2 + 4x + 5 = (x^2 + 4x + 4) + 1 = (x+2)^2 + 1$.
Then $\int \frac{dx}{x^2 + 4x + 5} = \int \frac{dx}{(x+2)^2 + 1} = \arctan(x+2) + C$.

**Solution 3**: Let $I = \int e^x \sin x \, dx$.
Using parts: $u = \sin x$, $dv = e^x \, dx$, $du = \cos x \, dx$, $v = e^x$.
$I = e^x \sin x - \int e^x \cos x \, dx$.
Now for $\int e^x \cos x \, dx$, let $u = \cos x$, $dv = e^x \, dx$, $du = -\sin x \, dx$, $v = e^x$.
$\int e^x \cos x \, dx = e^x \cos x + \int e^x \sin x \, dx = e^x \cos x + I$.
Substitute back: $I = e^x \sin x - (e^x \cos x + I) = e^x (\sin x - \cos x) - I$.
Thus $2I = e^x (\sin x - \cos x)$, so $I = \frac{e^x}{2}(\sin x - \cos x) + C$.

## Related Concepts

- **Differentiation**: The inverse operation of integration
- **Definite Integral**: The integral evaluated between two limits (MATH-062)
- **Indefinite Integral**: The antiderivative family (MATH-063)
- **Riemann Sum**: The limit definition of the definite integral
- **Fundamental Theorem of Calculus**: Links derivatives and integrals
- **Differential Equations**: Equations involving derivatives are solved using integration
- **Probability Density Functions**: Must integrate to 1 over their domain

## Next Concepts

- **Definite Integral**: Computing areas and net change with limits (MATH-062)
- **Indefinite Integral**: Antiderivatives and integration techniques (MATH-063)
- **Multiple Integrals**: Extending integration to higher dimensions (MATH-064)
- **Differential Equations**: Solving equations involving rates of change
- **Integral Transforms**: Laplace and Fourier transforms as generalised integrals

## Summary

Integral calculus is the branch of mathematics concerned with accumulation, area, and the inverse of differentiation. The indefinite integral $\int f(x) \, dx = F(x) + C$ yields a family of antiderivatives, while the definite integral $\int_a^b f(x) \, dx$ computes the signed area under a curve. Key techniques include the power rule, substitution, integration by parts, and partial fractions.

Integration is foundational to probability theory (normalising densities, expected values), machine learning (variational inference, entropy, kernel methods), and essentially all quantitative sciences. The Fundamental Theorem of Calculus unifies integration and differentiation, establishing them as inverse operations.

## Key Takeaways

- The integral is the inverse operation of differentiation (antiderivative)
- Indefinite integrals produce families of functions differing by a constant $C$
- Integration by substitution is the reverse chain rule
- Integration by parts is the reverse product rule
- The power rule fails for $n = -1$, requiring $\ln|x|$ instead
- Integration is essential for probability (normalisation, expectations, entropy)
- Many integrals do not have closed-form solutions and require numerical methods
- The Fundamental Theorem of Calculus connects derivatives and integrals
- Integration is a linear operation (sums and constant multiples)
- Mastery of basic integrals is essential for advanced machine learning theory
