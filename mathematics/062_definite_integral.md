# Concept: Definite Integral

## Concept ID

MATH-062

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the definite integral as the limit of Riemann sums
- State and apply the Fundamental Theorem of Calculus
- Compute definite integrals using antiderivatives and limit definitions
- Interpret the definite integral as signed area and net change
- Compute areas between curves and average values of functions
- Apply definite integrals to probability (expected value, normalisation)

## Prerequisites

- Differential calculus (derivatives, antiderivatives)
- Basic integral calculus (MATH-061)
- Limits and continuity
- Riemann sums
- Fundamental Theorem of Calculus

## Definition

Let $f$ be a function defined on the closed interval $[a, b]$. Partition $[a, b]$ into $n$ subintervals of equal width $\Delta x = \frac{b-a}{n}$, and choose sample points $x_i^*$ in each subinterval. The **definite integral** of $f$ from $a$ to $b$ is defined as the limit of the Riemann sum as $n \to \infty$:

$$
\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x
$$

provided this limit exists (in which case $f$ is said to be Riemann integrable on $[a, b]$).

By the **Fundamental Theorem of Calculus**, if $F$ is any antiderivative of $f$ (i.e., $F'(x) = f(x)$), then:

$$
\int_a^b f(x) \, dx = F(b) - F(a)
$$

## Intuition

The definite integral measures the total accumulation of a quantity over an interval. If $f(x)$ represents a rate of change (e.g., velocity), then $\int_a^b f(x) \, dx$ is the net change in the quantity (e.g., displacement) from $x = a$ to $x = b$.

Geometrically, imagine approximating the area under $y = f(x)$ by drawing $n$ rectangles of equal width and summing their areas. As $n$ increases, the rectangles become thinner, and the approximation improves. In the limit as the width goes to zero, the sum converges to the exact signed area between the curve and the x-axis.

The "signed" aspect is important: area above the x-axis contributes positively, while area below the x-axis contributes negatively. The definite integral gives the net area (algebraic sum), not the total area.

## Why This Concept Matters

The definite integral is one of the most widely used mathematical tools in science and engineering. It computes total quantities from rates of change: total distance from velocity, total work from force, total charge from current, total mass from density.

In probability and statistics, the definite integral is essential. Expected values, variances, cumulative distribution functions, and marginal probabilities are all defined through definite integrals. Every probability density function $p(x)$ must satisfy $\int_{-\infty}^{\infty} p(x) \, dx = 1$. The area under the receiver operating characteristic (ROC) curve, a key metric for classification models, is precisely the definite integral of the true positive rate against the false positive rate.

In machine learning, definite integrals appear in loss functions (expected risk), Bayesian inference (marginal likelihood, posterior expectations), and information theory (entropy, mutual information).

## Historical Background

The definite integral was formalised by **Bernhard Riemann** (1826–1866) in his 1854 habilitation thesis "On the Representation of a Function by a Trigonometric Series." Riemann's definition clarified the conditions under which a function could be integrated and laid the foundation for rigorous analysis.

Earlier, **Augustin-Louis Cauchy** (1789–1857) had defined definite integrals for continuous functions using limits of sums. Riemann extended this to a broader class of functions that could have many discontinuities, as long as the set of discontinuities was "small" in a precise sense.

The modern notation $\int_a^b f(x) \, dx$ was introduced by **Gottfried Wilhelm Leibniz** in the 17th century. The integral symbol $\int$ is an elongated S standing for _summa_ (Latin for "sum"), and the limits $a$ and $b$ indicate the boundaries of the interval.

**Isaac Newton** and Leibniz independently discovered the Fundamental Theorem of Calculus, which establishes the connection between the definite integral (as a limit of sums) and antiderivatives (the inverse of differentiation). This theorem made computing definite integrals practical, transforming calculus from a geometric curiosity into a powerful computational tool.

## Real World Examples

1. **Physics: Work Done by a Variable Force** — If a force $F(x)$ varies with position, the work done moving an object from $x=a$ to $x=b$ is $W = \int_a^b F(x) \, dx$. For example, the work required to stretch a spring follows Hooke's law $F = kx$, so $W = \int_0^d kx \, dx = \frac{1}{2}kd^2$.

2. **Medicine: Cardiac Output** — The total volume of blood pumped by the heart per minute is computed using the dye dilution method. The concentration of dye over time $c(t)$ is measured, and cardiac output is $Q = \frac{D}{\int_0^{\infty} c(t) \, dt}$, where $D$ is the amount of dye injected.

3. **Economics: Total Revenue and Cost** — If the marginal revenue $MR(q)$ is known (revenue per unit), then the total revenue from producing $q_1$ to $q_2$ units is $R = \int_{q_1}^{q_2} MR(q) \, dq$.

4. **Machine Learning: AUC-ROC** — The Area Under the ROC Curve is $\int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$, measuring classifier performance across all classification thresholds. An AUC of 1 indicates perfect classification.

5. **Environmental Science: Total Rainfall** — If $r(t)$ is the rainfall rate in cm/hour, the total rainfall from time $t_1$ to $t_2$ is $\int_{t_1}^{t_2} r(t) \, dt$.

## AI/ML Relevance

1. **Expected Value Computation** — The expected value of a continuous random variable is $\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) \, dx$, where $f_X$ is the probability density function. This is the defining integral of the mean and appears in every loss function expressed as expectation.

2. **Marginalisation** — Given a joint distribution $p(x, y)$, the marginal distribution is $p(x) = \int p(x, y) \, dy$. This integration eliminates (marginalises out) the variable $y$. Marginalisation is a core operation in Bayesian inference for latent variable models.

3. **Bayesian Model Evidence** — The marginal likelihood (model evidence) is $p(D) = \int p(D|\theta) p(\theta) \, d\theta$, where $\theta$ are parameters. This integral is used for Bayesian model comparison and is often the hardest computational challenge in Bayesian ML.

4. **Cumulative Distribution Functions** — The CDF $F(x) = \int_{-\infty}^x f(t) \, dt$ gives the probability that a random variable is less than or equal to $x$. The CDF is used for hypothesis testing, confidence intervals, and probability computations.

5. **ROC-AUC in Classification** — The AUC metric computes $\int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$ and is equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

6. **Bayesian Decision Theory** — Optimal decisions minimise expected risk: $R(a) = \int L(a, \theta) p(\theta|x) \, d\theta$, where $L$ is a loss function and $p(\theta|x)$ is the posterior.

## Mathematical Explanation

### Riemann Sum Definition

For a function $f$ on $[a, b]$, partition the interval as $a = x_0 < x_1 < \dots < x_n = b$ with $\Delta x_i = x_i - x_{i-1}$. Choose sample points $x_i^* \in [x_{i-1}, x_i]$. The Riemann sum is:

$$
S_n = \sum_{i=1}^n f(x_i^*) \Delta x_i
$$

The definite integral is the limit as the mesh (maximum subinterval width) goes to zero:

$$
\int_a^b f(x) \, dx = \lim_{\|\Delta\| \to 0} \sum_{i=1}^n f(x_i^*) \Delta x_i
$$

### Properties of Definite Integrals

1. **Reversal of Limits**: $\int_b^a f(x) \, dx = -\int_a^b f(x) \, dx$
2. **Zero Width**: $\int_a^a f(x) \, dx = 0$
3. **Linearity**: $\int_a^b [cf(x) + dg(x)] \, dx = c\int_a^b f(x) \, dx + d\int_a^b g(x) \, dx$
4. **Additivity**: $\int_a^b f(x) \, dx + \int_b^c f(x) \, dx = \int_a^c f(x) \, dx$
5. **Comparison**: If $f(x) \leq g(x)$ on $[a, b]$, then $\int_a^b f(x) \, dx \leq \int_a^b g(x) \, dx$
6. **Absolute Value**: $\left|\int_a^b f(x) \, dx\right| \leq \int_a^b |f(x)| \, dx$

### Fundamental Theorem of Calculus

**Part 1**: If $f$ is continuous on $[a, b]$, then $g(x) = \int_a^x f(t) \, dt$ is continuous on $[a, b]$, differentiable on $(a, b)$, and $g'(x) = f(x)$.

**Part 2**: If $F$ is any antiderivative of $f$, then $\int_a^b f(x) \, dx = F(b) - F(a)$.

### Area Between Curves

The area between $y = f(x)$ and $y = g(x)$ from $x = a$ to $x = b$, where $f(x) \geq g(x)$:

$$
A = \int_a^b [f(x) - g(x)] \, dx
$$

### Average Value of a Function

The average value of $f$ on $[a, b]$ is:

$$
f_{\text{avg}} = \frac{1}{b-a} \int_a^b f(x) \, dx
$$

## Formula(s)

1. **Definite Integral (Riemann)**:
   $$
   \int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^n f\left(a + i\frac{b-a}{n}\right) \frac{b-a}{n}
   $$

2. **Fundamental Theorem of Calculus, Part 2**:
   $$
   \int_a^b f(x) \, dx = F(b) - F(a) \quad \text{where } F'(x) = f(x)
   $$

3. **Area Between Curves**:
   $$
   A = \int_a^b |f(x) - g(x)| \, dx
   $$

4. **Average Value**:
   $$
   f_{\text{avg}} = \frac{1}{b-a} \int_a^b f(x) \, dx
   $$

5. **Integration by Parts for Definite Integrals**:
   $$
   \int_a^b u \, dv = [uv]_a^b - \int_a^b v \, du
   $$

6. **Substitution for Definite Integrals**:
   $$
   \int_a^b f(g(x)) g'(x) \, dx = \int_{g(a)}^{g(b)} f(u) \, du
   $$

## Properties

1. **Sign**: $\int_a^b f(x) \, dx$ can be positive, negative, or zero, representing net signed area.

2. **Monotonicity**: If $f$ is continuous and $f(x) \geq 0$ on $[a, b]$, then $\int_a^b f(x) \, dx \geq 0$, with equality iff $f \equiv 0$.

3. **Triangle Inequality**: $\left|\int_a^b f(x) \, dx\right| \leq \int_a^b |f(x)| \, dx$.

4. **Mean Value Theorem for Integrals**: There exists $c \in [a, b]$ such that $\int_a^b f(x) \, dx = f(c)(b-a)$.

5. **Differentiation Under the Integral Sign** (Leibniz rule):
   $$
   \frac{d}{dt} \int_{a(t)}^{b(t)} f(x, t) \, dx = f(b(t), t) b'(t) - f(a(t), t) a'(t) + \int_{a(t)}^{b(t)} \frac{\partial f}{\partial t} \, dx
   $$

6. **Change of Variables**: When substituting $u = g(x)$, the limits transform as $\int_a^b f(g(x))g'(x) \, dx = \int_{g(a)}^{g(b)} f(u) \, du$.

7. **Periodicity**: If $f$ is periodic with period $T$, then $\int_a^{a+T} f(x) \, dx$ is independent of $a$.

## Step-by-Step Worked Examples

### Example 1: Definite Integral via FTC

**Problem**: Compute $\int_1^3 (2x^2 - 3x + 1) \, dx$.

**Solution**:

Step 1: Find an antiderivative.

$$
F(x) = \int (2x^2 - 3x + 1) \, dx = \frac{2x^3}{3} - \frac{3x^2}{2} + x
$$

Step 2: Apply FTC Part 2.

$$
\int_1^3 (2x^2 - 3x + 1) \, dx = F(3) - F(1)
$$

Step 3: Evaluate $F(3)$.

$$
F(3) = \frac{2(27)}{3} - \frac{3(9)}{2} + 3 = 18 - \frac{27}{2} + 3 = 21 - 13.5 = 7.5
$$

Step 4: Evaluate $F(1)$.

$$
F(1) = \frac{2}{3} - \frac{3}{2} + 1 = \frac{4}{6} - \frac{9}{6} + \frac{6}{6} = \frac{1}{6}
$$

Step 5: Compute the difference.

$$
F(3) - F(1) = 7.5 - \frac{1}{6} = \frac{45}{6} - \frac{1}{6} = \frac{44}{6} = \frac{22}{3}
$$

Result: $\int_1^3 (2x^2 - 3x + 1) \, dx = \frac{22}{3}$.

### Example 2: Definite Integral with Substitution

**Problem**: Compute $\int_0^2 x e^{x^2} \, dx$.

**Solution**:

Step 1: Let $u = x^2$, so $du = 2x \, dx$, giving $x \, dx = \frac{du}{2}$.

Step 2: Change the limits. When $x = 0$, $u = 0$. When $x = 2$, $u = 4$.

Step 3: Rewrite and evaluate.

$$
\int_0^2 x e^{x^2} \, dx = \int_{u=0}^{4} e^u \cdot \frac{du}{2} = \frac{1}{2} \int_0^4 e^u \, du
$$

$$
= \frac{1}{2} [e^u]_0^4 = \frac{1}{2} (e^4 - e^0) = \frac{1}{2} (e^4 - 1)
$$

Result: $\int_0^2 x e^{x^2} \, dx = \frac{e^4 - 1}{2}$.

### Example 3: Area Between Curves

**Problem**: Find the area between $y = x^2$ and $y = x + 2$.

**Solution**:

Step 1: Find intersection points.

$x^2 = x + 2 \implies x^2 - x - 2 = 0 \implies (x-2)(x+1) = 0$, so $x = -1$ and $x = 2$.

Step 2: Determine which function is on top on $[-1, 2]$.

At $x = 0$: $x^2 = 0$, $x + 2 = 2$. So $y = x + 2$ is above $y = x^2$.

Step 3: Set up and evaluate the integral.

$$
A = \int_{-1}^2 [(x+2) - x^2] \, dx = \int_{-1}^2 (-x^2 + x + 2) \, dx
$$

$$
= \left[-\frac{x^3}{3} + \frac{x^2}{2} + 2x\right]_{-1}^2
$$

At $x = 2$: $-\frac{8}{3} + \frac{4}{2} + 4 = -\frac{8}{3} + 2 + 4 = -\frac{8}{3} + 6 = \frac{10}{3}$

At $x = -1$: $-(-\frac{1}{3}) + \frac{1}{2} - 2 = \frac{1}{3} + \frac{1}{2} - 2 = \frac{2}{6} + \frac{3}{6} - \frac{12}{6} = -\frac{7}{6}$

$$
A = \frac{10}{3} - \left(-\frac{7}{6}\right) = \frac{10}{3} + \frac{7}{6} = \frac{20}{6} + \frac{7}{6} = \frac{27}{6} = \frac{9}{2}
$$

The area is $\frac{9}{2}$ square units.

### Example 4: Expected Value of a Continuous Random Variable

**Problem**: Let $X$ be a continuous random variable with PDF $f(x) = 2e^{-2x}$ for $x \geq 0$ (exponential distribution with rate $\lambda = 2$). Compute $\mathbb{E}[X] = \int_0^{\infty} x f(x) \, dx$.

**Solution**:

Step 1: Write the integral.

$$
\mathbb{E}[X] = \int_0^{\infty} x \cdot 2e^{-2x} \, dx
$$

Step 2: Integrate by parts. Let $u = x$, $dv = 2e^{-2x} \, dx$, so $du = dx$, $v = -e^{-2x}$.

$$
\mathbb{E}[X] = \left[-x e^{-2x}\right]_0^{\infty} + \int_0^{\infty} e^{-2x} \, dx
$$

Step 3: Evaluate the boundary term. As $x \to \infty$, $x e^{-2x} \to 0$ (exponential dominates). At $x = 0$, $-0 \cdot e^0 = 0$. So $[-x e^{-2x}]_0^{\infty} = 0 - 0 = 0$.

Step 4: Evaluate the remaining integral.

$$
\mathbb{E}[X] = \int_0^{\infty} e^{-2x} \, dx = \left[-\frac{1}{2} e^{-2x}\right]_0^{\infty} = 0 - \left(-\frac{1}{2}\right) = \frac{1}{2}
$$

The expected value is $\frac{1}{2}$, matching the known formula $\mathbb{E}[X] = 1/\lambda$ for exponential distributions.

### Example 5: AUC Computation

**Problem**: A classifier produces scores such that positive examples have scores distributed as $\text{Exp}(1)$ shifted by $+1$, and negatives as $\text{Exp}(1)$. The ROC curve parameterises TPR and FPR as functions of threshold $t$. For threshold $t$, $\text{TPR}(t) = e^{-(t-1)}$ for $t \geq 1$ and $\text{FPR}(t) = e^{-t}$ for $t \geq 0$. Compute the AUC by integrating $\text{TPR}$ as a function of $\text{FPR}$.

**Solution**:

Step 1: Express TPR in terms of FPR. We have $\text{FPR} = e^{-t}$, so $t = -\ln(\text{FPR})$. Then $\text{TPR} = e^{-(t-1)} = e^{-t} e^1 = e \cdot \text{FPR}$, valid for $t \geq 1$ i.e., $\text{FPR} \leq e^{-1}$.

For $\text{FPR} > e^{-1}$ ($t < 1$), $\text{TPR} = 1$ (since all positives have score $\geq 1$).

Step 2: Split the integral.

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR}) = \int_0^{e^{-1}} e \cdot \text{FPR} \, d(\text{FPR}) + \int_{e^{-1}}^1 1 \, d(\text{FPR})
$$

Step 3: Evaluate.

$$
= \left[\frac{e}{2} \cdot \text{FPR}^2\right]_0^{e^{-1}} + [\text{FPR}]_{e^{-1}}^1 = \frac{e}{2} \cdot e^{-2} + (1 - e^{-1}) = \frac{1}{2e} + 1 - \frac{1}{e} = 1 - \frac{1}{2e}
$$

AUC $\approx 1 - \frac{1}{2e} \approx 0.816$, better than random (0.5) but not perfect.

## Visual Interpretation

The definite integral $\int_a^b f(x) \, dx$ can be visualised as the signed area under the curve $y = f(x)$ between $x = a$ and $x = b$. This area is built from infinitesimally thin vertical rectangles of height $f(x)$ and width $dx$.

For non-negative $f$, the integral is the total area under the curve. For functions that cross the x-axis, the integral computes the net area: regions above the axis contribute positively, regions below contribute negatively. To compute total (unsigned) area, integrate $|f(x)|$ instead.

The Riemann sum visualisation is powerful: draw $n$ rectangles of equal width $\Delta x = (b-a)/n$, each with height $f(x_i^*)$ where $x_i^*$ is a point in the $i$-th subinterval. As $n$ grows, the rectangle tops approximate the curve more closely, and the sum converges to the integral.

The average value interpretation is another useful visualisation: $f_{\text{avg}}$ is the height of a rectangle of width $b-a$ that has the same area as the region under the curve.

## Common Mistakes

1. **Forgetting to change limits with substitution**: When performing substitution on a definite integral, the limits must be transformed along with the integrand. A common error is to compute the antiderivative in $u$ but evaluate at the original $x$ limits.

2. **Misapplying the Fundamental Theorem**: The FTC requires that the antiderivative be differentiable (and thus continuous) on $[a, b]]. If $F$ has a discontinuity (e.g., a vertical asymptote), the FTC does not apply directly.

3. **Confusing net area with total area**: $\int_a^b f(x) \, dx$ gives signed area. For total area, use $\int_a^b |f(x)| \, dx$. This distinction is critical in applications: total distance travelled is $\int |v(t)| \, dt$, not $\int v(t) \, dt$.

4. **Sign errors when reversing limits**: $\int_b^a f(x) \, dx = -\int_a^b f(x) \, dx$. Forgetting the minus sign is a common algebraic mistake.

5. **Improper handling of improper integrals**: Integrals with infinite limits or integrand singularities require limit definitions. Evaluating $\int_0^{\infty} f(x) \, dx$ as if it were a standard definite integral can lead to incorrect results.

6. **Assuming all functions are integrable**: Not all functions are Riemann integrable (e.g., Dirichlet's function). The function must be bounded and its set of discontinuities must have measure zero.

7. **Incorrectly applying additivity**: $\int_a^b f + \int_b^c f = \int_a^c f$ holds only if the integrals exist. With improper integrals, this requires careful handling.

8. **Forgetting the $dx$**: The differential $dx$ in the integral notation $\int_a^b f(x) \, dx$ indicates the variable of integration. Omitting it is a syntax error and can lead to confusion in multivariable settings.

## Interview Questions

### Beginner

1. **Q**: What is the Fundamental Theorem of Calculus?
   **A**: It connects differentiation and integration. Part 1: if $g(x) = \int_a^x f(t) \, dt$, then $g'(x) = f(x)$. Part 2: $\int_a^b f(x) \, dx = F(b) - F(a)$ where $F'(x) = f(x)$.

2. **Q**: Compute $\int_0^1 x^3 \, dx$.
   **A**: $\left[\frac{x^4}{4}\right]_0^1 = \frac{1}{4}$.

3. **Q**: What is the difference between $\int_a^b f(x) \, dx$ and $\int f(x) \, dx$?
   **A**: $\int_a^b f(x) \, dx$ is a definite integral (a number representing signed area), while $\int f(x) \, dx$ is an indefinite integral (a family of antiderivative functions $F(x) + C$).

4. **Q**: Evaluate $\int_0^{\pi} \sin x \, dx$.
   **A**: $[-\cos x]_0^{\pi} = -\cos\pi - (-\cos 0) = -(-1) - (-1) = 1 + 1 = 2$.

5. **Q**: Compute the average value of $f(x) = 4x$ on $[0, 2]$.
   **A**: $f_{\text{avg}} = \frac{1}{2-0} \int_0^2 4x \, dx = \frac{1}{2} [2x^2]_0^2 = \frac{1}{2} \cdot 8 = 4$.

### Intermediate

1. **Q**: Compute $\int_{-1}^1 |x| \, dx$.
   **A**: Split at $x=0$: $\int_{-1}^0 (-x) \, dx + \int_0^1 x \, dx = [-\frac{x^2}{2}]_{-1}^0 + [\frac{x^2}{2}]_0^1 = (0 + \frac{1}{2}) + (\frac{1}{2} - 0) = 1$.

2. **Q**: Find the area between $y = x^2$ and $y = x^3$ for $x \in [0, 1]$.
   **A**: $x^2 \geq x^3$ on $[0, 1]$. Area $= \int_0^1 (x^2 - x^3) \, dx = [\frac{x^3}{3} - \frac{x^4}{4}]_0^1 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}$.

3. **Q**: How is the definite integral used to compute expected values in probability?
   **A**: $\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) f_X(x) \, dx$, where $f_X$ is the PDF. For $\mathbb{E}[X]$, $g(x) = x$. For variance, $\mathbb{E}[(X-\mu)^2]$.

4. **Q**: Compute $\int_0^{\infty} e^{-3x} \, dx$.
   **A**: $\lim_{b \to \infty} \int_0^b e^{-3x} \, dx = \lim_{b \to \infty} [-\frac{1}{3} e^{-3x}]_0^b = \lim_{b \to \infty} (-\frac{1}{3}e^{-3b} + \frac{1}{3}) = \frac{1}{3}$.

5. **Q**: What is the Leibniz rule for differentiating under the integral sign? Give an ML example of its use.
   **A**: $\frac{d}{dt} \int_{a(t)}^{b(t)} f(x,t) \, dx = f(b(t),t)b'(t) - f(a(t),t)a'(t) + \int_{a(t)}^{b(t)} \frac{\partial f}{\partial t} \, dx$. In ML, this is used to compute gradients of the ELBO in variational inference through the reparameterisation trick.

### Advanced

1. **Q**: Compute $\int_0^{\infty} x^2 e^{-x} \, dx$ (Gamma function).
   **A**: This is $\Gamma(3) = 2! = 2$. Using parts: let $u = x^2$, $dv = e^{-x}dx$, then $du = 2x\,dx$, $v = -e^{-x}$. $\int_0^{\infty} x^2 e^{-x} \, dx = [-x^2 e^{-x}]_0^{\infty} + 2\int_0^{\infty} x e^{-x} \, dx = 0 + 2[ -x e^{-x}]_0^{\infty} + 2\int_0^{\infty} e^{-x} \, dx = 2$. The Gamma function $\Gamma(n) = \int_0^{\infty} x^{n-1} e^{-x} \, dx = (n-1)!$ for integers $n$.

2. **Q**: Explain how Monte Carlo integration uses definite integrals and why it is important for Bayesian machine learning.
   **A**: Monte Carlo integration estimates $\int f(x)p(x) \, dx \approx \frac{1}{N}\sum_{i=1}^N f(x_i)$ where $x_i \sim p(x)$. It is crucial for Bayesian ML because posterior expectations $\mathbb{E}_{p(\theta|D)}[g(\theta)] = \int g(\theta) p(\theta|D) \, d\theta$ are typically intractable analytically. Monte Carlo methods (MCMC, importance sampling, SMC) provide numerical approximations using samples.

3. **Q**: Derive the bias-variance decomposition of the mean squared error as integrals involving the true function, the estimator, and the noise distribution.
   **A**: For estimator $\hat{f}(x)$ trained on data $D$, the expected prediction error at point $x$ is:
   $$
   \mathbb{E}_D[(y - \hat{f}(x))^2] = \mathbb{E}_D[(f(x) + \epsilon - \hat{f}(x))^2]
   $$
   where $y = f(x) + \epsilon$ with $\mathbb{E}[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$.
   Expanding: $= \mathbb{E}_D[(f(x) - \hat{f}(x))^2] + \sigma^2$ (since $\epsilon$ is independent).
   Then $\mathbb{E}_D[(f - \hat{f})^2] = (f - \mathbb{E}[\hat{f}])^2 + \mathbb{E}_D[(\hat{f} - \mathbb{E}[\hat{f}])^2] = \text{Bias}^2 + \text{Variance}$.
   The expected risk (integrated MSE) is the integral over $x$ of this decomposition, weighted by the input distribution $p(x)$: $\int (\text{Bias}^2(x) + \text{Variance}(x) + \sigma^2) p(x) \, dx$.

## Practice Problems

### Easy

1. Compute $\int_0^2 (6x^2 - 2x) \, dx$.

2. Evaluate $\int_1^e \frac{1}{x} \, dx$.

3. Compute $\int_0^{\pi/2} \cos x \, dx$.

4. Find the area under $y = e^x$ from $x = 0$ to $x = 1$.

5. Compute $\int_{-1}^2 3 \, dx$.

### Medium

1. Use substitution to compute $\int_0^1 \frac{2x}{1 + x^2} \, dx$.

2. Compute the area between $y = 4 - x^2$ and $y = 0$ (the x-axis).

3. Evaluate $\int_0^2 x e^{x^2} \, dx$ using substitution.

4. Find the average value of $f(x) = \sin x$ on $[0, \pi]$.

5. Compute $\int_0^1 x(1-x)^3 \, dx$ (hint: use substitution $u = 1-x$).

### Hard

1. Compute $\int_0^{\infty} x^3 e^{-x^2} \, dx$ (hint: use $u = x^2$).

2. Find the area between $y = \sin x$ and $y = \cos x$ over one period where $\sin x \geq \cos x$, specifically from $x = \pi/4$ to $x = 5\pi/4$.

3. The logistic sigmoid is $\sigma(x) = \frac{1}{1+e^{-x}}$. Compute $\int_{-\infty}^{\infty} \sigma(x)(1 - \sigma(x)) \, dx$. (Hint: note that $\sigma'(x) = \sigma(x)(1-\sigma(x))$.)

## Solutions

### Easy Solutions

**Solution 1**: $\int_0^2 (6x^2 - 2x) \, dx = [2x^3 - x^2]_0^2 = (16 - 4) - 0 = 12$.

**Solution 2**: $\int_1^e \frac{1}{x} \, dx = [\ln x]_1^e = \ln e - \ln 1 = 1 - 0 = 1$.

**Solution 3**: $\int_0^{\pi/2} \cos x \, dx = [\sin x]_0^{\pi/2} = 1 - 0 = 1$.

**Solution 4**: $\int_0^1 e^x \, dx = [e^x]_0^1 = e^1 - e^0 = e - 1$.

**Solution 5**: $\int_{-1}^2 3 \, dx = [3x]_{-1}^2 = 6 - (-3) = 9$.

### Medium Solutions

**Solution 1**: Let $u = 1 + x^2$, $du = 2x \, dx$. Limits: $x=0 \to u=1$, $x=1 \to u=2$.
$\int_0^1 \frac{2x}{1+x^2} \, dx = \int_1^2 \frac{du}{u} = [\ln u]_1^2 = \ln 2$.

**Solution 2**: Intersection: $4 - x^2 = 0 \implies x = \pm 2$.
$A = \int_{-2}^2 (4 - x^2) \, dx = [4x - \frac{x^3}{3}]_{-2}^2 = (8 - \frac{8}{3}) - (-8 + \frac{8}{3}) = \frac{16}{3} + \frac{16}{3} = \frac{32}{3}$.

**Solution 3**: $u = x^2$, $du = 2x\,dx$, $x\,dx = du/2$. Limits: $x=0 \to u=0$, $x=2 \to u=4$.
$\int_0^2 x e^{x^2} \, dx = \frac{1}{2} \int_0^4 e^u \, du = \frac{1}{2}[e^u]_0^4 = \frac{e^4 - 1}{2}$.

**Solution 4**: $f_{\text{avg}} = \frac{1}{\pi} \int_0^{\pi} \sin x \, dx = \frac{1}{\pi}[-\cos x]_0^{\pi} = \frac{1}{\pi}(1 + 1) = \frac{2}{\pi}$.

**Solution 5**: Let $u = 1 - x$, $du = -dx$, limits: $x=0 \to u=1$, $x=1 \to u=0$.
$\int_0^1 x(1-x)^3 \, dx = \int_1^0 (1-u) u^3 (-du) = \int_0^1 (u^3 - u^4) \, du = [\frac{u^4}{4} - \frac{u^5}{5}]_0^1 = \frac{1}{4} - \frac{1}{5} = \frac{1}{20}$.

### Hard Solutions

**Solution 1**: Let $u = x^2$, $du = 2x\,dx$, $x\,dx = du/2$. Also $x^3 = x \cdot x^2 = xu$.
$\int_0^{\infty} x^3 e^{-x^2} \, dx = \int_0^{\infty} x \cdot x^2 \cdot e^{-x^2} \, dx = \frac{1}{2} \int_0^{\infty} u e^{-u} \, du = \frac{1}{2} \Gamma(2) = \frac{1}{2} \cdot 1! = \frac{1}{2}$.

**Solution 2**: Area $= \int_{\pi/4}^{5\pi/4} (\sin x - \cos x) \, dx = [-\cos x - \sin x]_{\pi/4}^{5\pi/4}$.
At $5\pi/4$: $\sin(5\pi/4) = -\sqrt{2}/2$, $\cos(5\pi/4) = -\sqrt{2}/2$. Value $= -(-\sqrt{2}/2) - (-\sqrt{2}/2) = \sqrt{2}/2 + \sqrt{2}/2 = \sqrt{2}$.
At $\pi/4$: $\sin(\pi/4) = \sqrt{2}/2$, $\cos(\pi/4) = \sqrt{2}/2$. Value $= -\sqrt{2}/2 - \sqrt{2}/2 = -\sqrt{2}$.
Area $= \sqrt{2} - (-\sqrt{2}) = 2\sqrt{2}$.

**Solution 3**: Note that $\sigma'(x) = \sigma(x)(1-\sigma(x)) = \frac{e^{-x}}{(1+e^{-x})^2}$.
Thus $\int_{-\infty}^{\infty} \sigma(x)(1-\sigma(x)) \, dx = \int_{-\infty}^{\infty} \sigma'(x) \, dx = \lim_{x \to \infty} \sigma(x) - \lim_{x \to -\infty} \sigma(x) = 1 - 0 = 1$.
This integral equals 1, demonstrating that the derivative of the sigmoid integrates to the total change in the sigmoid function.

## Related Concepts

- **Indefinite Integral**: The antiderivative family, related by the FTC (MATH-063)
- **Riemann Sum**: The limit definition of the definite integral
- **Fundamental Theorem of Calculus**: Links definite and indefinite integrals
- **Improper Integrals**: Definite integrals with infinite limits or singularities
- **Multiple Integrals**: Generalisation to higher dimensions (MATH-064)
- **Probability Density Functions**: Must integrate to 1 (normalisation)
- **Expected Value**: Defined as an integral of $x p(x)$

## Next Concepts

- **Indefinite Integral**: Antiderivatives without limits (MATH-063)
- **Multiple Integrals**: Integrating over 2D and 3D regions (MATH-064)
- **Improper Integrals**: Definite integrals with unbounded domains or integrands
- **Integral Transforms**: Laplace transforms, Fourier transforms
- **Numerical Integration**: Approximating definite integrals when closed forms are unavailable

## Summary

The definite integral $\int_a^b f(x) \, dx$ represents the signed area under $f$ from $a$ to $b$, defined as the limit of Riemann sums. The Fundamental Theorem of Calculus provides the practical computational tool: evaluate an antiderivative at the limits and subtract. Definite integrals are used to compute areas, volumes, average values, and accumulated change.

In machine learning, definite integrals are fundamental to probability theory (normalisation, expected values, marginalisation), Bayesian inference (model evidence, posterior expectations), and model evaluation (AUC). They appear in virtually every ML loss function expressed as an expectation.

## Key Takeaways

- The definite integral computes signed area through the limit of Riemann sums
- The FTC enables evaluation via antiderivatives: $\int_a^b f = F(b) - F(a)$
- Substitution in definite integrals requires transforming the limits
- The definite integral gives net (signed) area; use $|f(x)|$ for total area
- Probability densities must integrate to 1: $\int_{-\infty}^{\infty} p(x) \, dx = 1$
- Expected values are integrals: $\mathbb{E}[g(X)] = \int g(x) p(x) \, dx$
- Marginalisation eliminates variables through integration: $p(x) = \int p(x,y) \, dy$
- AUC integrates TPR against FPR over all thresholds
- The Leibniz rule differentiates under the integral sign
- The average value of $f$ on $[a, b]$ is $\frac{1}{b-a}\int_a^b f(x) \, dx$
