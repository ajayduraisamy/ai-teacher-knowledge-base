# Concept: Limits

## Concept ID

MATH-053

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the limit of a function and interpret it both informally and formally using the epsilon-delta ($\varepsilon$-$\delta$) definition.
- Evaluate limits analytically, numerically, and graphically, including one-sided limits at critical points.
- Apply limit laws (sum, product, quotient, power, composition) to compute limits of algebraic and transcendental functions.
- Compute $\lim_{x\to 0} \frac{\sin x}{x} = 1$ and $\lim_{n\to\infty} (1 + 1/n)^n = e$, two foundational limits.
- Recognize and resolve indeterminate forms ($0/0$, $\infty/\infty$, $0\cdot\infty$, $\infty - \infty$, $0^0$, $1^\infty$, $\infty^0$) using algebraic manipulation and L'Hôpital's rule.
- Connect limits to convergence analysis, numerical stability, and gradient dynamics in AI/ML training.

## Prerequisites

- Functions (MATH-044), domain (MATH-045), range (MATH-046), and composite functions (MATH-047).
- Polynomial (MATH-049), exponential (MATH-050), logarithmic (MATH-051), and trigonometric functions (MATH-052).
- Basic algebra of rational expressions, factoring, and rationalisation.

## Definition

The **limit** of a function $f(x)$ as $x$ approaches a value $a$ is the value that $f(x)$ gets arbitrarily close to as $x$ gets arbitrarily close to $a$. Formally, we write:

$$\lim_{x \to a} f(x) = L$$

which is read as "the limit of $f(x)$ as $x$ approaches $a$ is $L$."

**Epsilon-Delta Definition:** For every $\varepsilon > 0$, there exists a $\delta > 0$ such that if $0 < |x - a| < \delta$, then $|f(x) - L| < \varepsilon$. This formalises the notion of "arbitrarily close": no matter how small a tolerance $\varepsilon$ we choose, we can find a $\delta$-neighbourhood around $a$ (excluding $a$ itself) where $f(x)$ stays within $\varepsilon$ of $L$.

**One-Sided Limits:**

$$\lim_{x \to a^-} f(x) \quad \text{(left-hand limit: $x$ approaches $a$ from below)}$$
$$\lim_{x \to a^+} f(x) \quad \text{(right-hand limit: $x$ approaches $a$ from above)}$$

The two-sided limit $\lim_{x\to a} f(x)$ exists if and only if both one-sided limits exist and are equal: $\lim_{x\to a^-} f(x) = \lim_{x\to a^+} f(x)$.

**Infinite Limits:** If $f(x)$ grows without bound as $x\to a$, we write $\lim_{x\to a} f(x) = \infty$ (or $-\infty$). This indicates the function has a vertical asymptote at $x = a$.

**Limits at Infinity:** If $f(x)$ approaches a finite value $L$ as $x \to \infty$ (or $x \to -\infty$), we write $\lim_{x\to\infty} f(x) = L$, indicating a horizontal asymptote $y = L$.

## Intuition

Imagine walking along the graph of $f(x)$ toward the vertical line $x = a$ from either direction. The limit $L$ is the $y$-value your shadow on the $y$-axis approaches as you get closer and closer to $a$, regardless of whether $f(a)$ actually equals $L$, or even whether $f(a)$ exists.

The critical insight is that limits describe **behaviour near a point**, not at the point itself. A function can have a well-defined limit at $a$ even if it is undefined at $a$. For example, $f(x) = \frac{x^2 - 1}{x - 1}$ is undefined at $x = 1$, yet $\lim_{x\to 1} f(x) = 2$ because $f(x) = x + 1$ for all $x \neq 1$.

For limits at infinity, imagine compressing the entire $x$-axis: as you zoom out further and further, the function's graph settles into a horizontal line $y = L$, provided the limit exists.

## Why This Concept Matters

Limits are the foundational concept of all calculus. Without limits, there is no definition of continuity, no derivative, no integral, no infinite series.

1. **Derivative Definition:** $f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}$ is a limit of a difference quotient — every derivative is a limit.

2. **Integral Definition:** $\int_a^b f(x) dx = \lim_{n\to\infty} \sum_{i=1}^n f(x_i^*) \Delta x$ — every definite integral is a limit of Riemann sums.

3. **Convergence of Sequences and Series:** The sum of an infinite series $\sum_{n=1}^\infty a_n$ is defined as the limit of its partial sums. Convergence of iterative algorithms (gradient descent, Newton's method) is analysed through limits.

4. **Numerical Stability:** Understanding limits helps explain why certain numerical computations (e.g., dividing by very small numbers) are unstable, which is crucial for implementing robust machine learning algorithms.

5. **Gradient Analysis:** The behaviour of gradients near critical points (saddle points, local minima) is analysed through limits, helping us understand why optimisation algorithms behave as they do.

## Historical Background

The concept of a limit has ancient roots. Eudoxus of Cnidus (c. 390-337 BCE) developed the method of exhaustion for calculating areas and volumes, which implicitly used limiting processes. Archimedes (c. 287-212 BCE) refined this method to compute areas under parabolas and approximate $\pi$.

In the 17th century, Newton and Leibniz independently developed calculus, but their use of "infinitesimals" — infinitely small quantities — was controversial. Bishop Berkeley famously criticised them as "ghosts of departed quantities." Despite the lack of rigour, calculus proved extraordinarily useful.

The rigorous foundation of limits came in the 19th century. Augustin-Louis Cauchy (1789-1857) gave the first clear definition of a limit in his *Cours d'Analyse* (1821). Karl Weierstrass (1815-1897) later formalised the $\varepsilon$-$\delta$ definition that is standard today. This rigorous foundation — called "arithmetisation of analysis" — resolved centuries of confusion and placed calculus on solid logical ground.

## Real World Examples

**Example 1: Instantaneous Velocity.** A car's position is $s(t) = t^2$ meters at time $t$ seconds. The average velocity between $t$ and $t+h$ is $\frac{(t+h)^2 - t^2}{h} = 2t + h$. The instantaneous velocity at $t$ is $\lim_{h\to 0} (2t + h) = 2t$ meters per second.

**Example 2: Compound Interest.** If you invest $P$ dollars at annual interest rate $r$ compounded $n$ times per year, after $t$ years you have $A = P(1 + r/n)^{nt}$. **Continuously compounded** interest is the limit as $n \to \infty$: $A = P e^{rt}$, which follows from $\lim_{n\to\infty} (1 + 1/n)^n = e$.

**Example 3: Terminal Velocity.** A skydiver falling experiences air resistance proportional to velocity squared: $v'(t) = g - kv^2$. As $t\to\infty$, velocity approaches terminal velocity $v_{\text{term}} = \sqrt{g/k}$, which is $\lim_{t\to\infty} v(t)$.

**Example 4: Population Growth.** A population modelled by logistic growth $P(t) = \frac{K}{1 + Ce^{-rt}}$ has carrying capacity $K = \lim_{t\to\infty} P(t)$. The limit represents the maximum sustainable population.

**Example 5: Drug Concentration.** The concentration of a drug in the bloodstream after repeated doses approaches a steady state: $\lim_{n\to\infty} C_n = \frac{C_0}{1 - e^{-kT}}$, where $C_0$ is the dose concentration, $k$ is the elimination rate, and $T$ is the dosing interval.

## AI/ML Relevance

Limits are essential to understanding the theoretical foundations of machine learning:

**1. Convergence Analysis of Gradient Descent.** Gradient descent iterates $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$. For convex loss functions, the iterates converge to the global minimum: $\lim_{t\to\infty} \theta_t = \theta^*$. The rate of convergence (linear, superlinear, quadratic) is analysed through limits of error ratios:

$$\lim_{t\to\infty} \frac{\|\theta_{t+1} - \theta^*\|}{\|\theta_t - \theta^*\|} = \rho$$

If $0 < \rho < 1$, convergence is linear; if $\rho = 0$, it is superlinear.

**2. Numerical Stability of Neural Network Training.** When computing gradients, small denominators can cause numerical instability. For instance, in batch normalisation, the variance estimate $\sigma^2$ is computed and then inverted: $1/\sigma^2$. If $\sigma^2 \to 0$ (e.g., during training collapse), $1/\sigma^2 \to \infty$, causing numerical overflow. Understanding limits helps engineers add small epsilon terms ($1/(\sigma^2 + \varepsilon)$) to prevent division by values approaching zero.

**3. Vanishing and Exploding Gradients.** In deep networks, gradients are products of many terms. Consider a simple $L$-layer network: $\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \prod_{i=2}^{L} W^{(i)} \cdot \prod_{i=1}^{L} \sigma'(z^{(i)})$. As $L \to \infty$:
- If each term $< 1$, the product tends to $0$ (vanishing gradients).
- If each term $> 1$, the product tends to $\infty$ (exploding gradients).
These are limits of products, and they explain why deep networks are hard to train without careful initialisation (Xavier, He initialisation) and normalisation techniques.

**4. Learning Rate Schedules.** A learning rate schedule $\alpha_t$ often satisfies $\sum_{t=1}^\infty \alpha_t = \infty$ and $\sum_{t=1}^\infty \alpha_t^2 < \infty$ (Robbins-Monro conditions for stochastic approximation). These conditions involve limits of infinite series and guarantee almost-sure convergence of SGD.

**5. Definition of the Derivative in Backpropagation.** Backpropagation fundamentally relies on the chain rule, whose proof depends on limits. The derivative $\frac{\partial L}{\partial w} = \lim_{\varepsilon\to 0} \frac{L(w+\varepsilon) - L(w)}{\varepsilon}$ tells us how much the loss changes when we nudge a weight by an infinitesimal amount.

**6. Loss Function Behaviour at Boundaries.** In classification, the cross-entropy loss $-\log(p)$ has $\lim_{p\to 0^+} -\log(p) = \infty$. This explains why confident wrong predictions are heavily penalised, driving the model to assign high probability to the correct class.

**7. Softmax Function.** The softmax function $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ involves limits: as temperature $T \to 0^+$ in $p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$, the softmax approaches a one-hot distribution (argmax), while as $T \to \infty$, it approaches a uniform distribution.

**8. Universal Approximation Theorem.** The universal approximation theorem states that a feedforward network with a single hidden layer can approximate any continuous function on a compact set arbitrarily well — this is a statement about limits of function sequences: $\lim_{n\to\infty} \|f_n - f\|_\infty = 0$, where $f_n$ is the network output with $n$ hidden units.

## Mathematical Explanation

**Limit Laws:** If $\lim_{x\to a} f(x) = L$ and $\lim_{x\to a} g(x) = M$, then:

1. **Sum:** $\lim_{x\to a} [f(x) + g(x)] = L + M$
2. **Difference:** $\lim_{x\to a} [f(x) - g(x)] = L - M$
3. **Product:** $\lim_{x\to a} [f(x) \cdot g(x)] = L \cdot M$
4. **Quotient:** $\lim_{x\to a} \frac{f(x)}{g(x)} = \frac{L}{M}$ provided $M \neq 0$
5. **Constant Multiple:** $\lim_{x\to a} [c \cdot f(x)] = c \cdot L$
6. **Power:** $\lim_{x\to a} [f(x)]^n = L^n$ for positive integer $n$
7. **Root:** $\lim_{x\to a} \sqrt[n]{f(x)} = \sqrt[n]{L}$ provided $L \geq 0$ for even $n$
8. **Composition:** If $\lim_{x\to a} g(x) = b$ and $\lim_{y\to b} f(y) = L$ with $f$ continuous at $b$, then $\lim_{x\to a} f(g(x)) = L$

**Squeeze (Sandwich) Theorem:** If $g(x) \leq f(x) \leq h(x)$ for all $x$ in an interval containing $a$ (except possibly at $a$) and $\lim_{x\to a} g(x) = \lim_{x\to a} h(x) = L$, then $\lim_{x\to a} f(x) = L$.

**Two Special Limits:**

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$
$$\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e$$

The first limit is proved using the squeeze theorem with geometric arguments on the unit circle. The second limit defines the number $e \approx 2.71828$.

**Indeterminate Forms and L'Hôpital's Rule.** For limits of the form $0/0$ or $\infty/\infty$:

$$\lim_{x\to a} \frac{f(x)}{g(x)} = \lim_{x\to a} \frac{f'(x)}{g'(x)}$$

provided the limit of the ratio of derivatives exists (or is $\pm\infty$). This rule can be applied repeatedly if the indeterminate form persists.

**Other Indeterminate Forms** can be converted to $0/0$ or $\infty/\infty$:
- $0 \cdot \infty$: Rewrite as $\frac{f}{1/g}$ or $\frac{g}{1/f}$
- $\infty - \infty$: Factor or combine fractions
- $0^0, 1^\infty, \infty^0$: Use $\ln$ to bring the exponent down: $f^g = e^{g \ln f}$

## Formula(s)

**Limit Definition (Derivative):**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Limit Definition (Integral):**
$$\int_a^b f(x) dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x$$

**Special Limits:**
$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$
$$\lim_{x \to 0} \frac{\cos x - 1}{x} = 0$$
$$\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e$$
$$\lim_{x \to 0} (1 + x)^{1/x} = e$$
$$\lim_{x \to \infty} \frac{\ln x}{x} = 0$$
$$\lim_{x \to \infty} \frac{x^p}{e^x} = 0 \quad \text{for any } p > 0$$

**L'Hôpital's Rule:**
$$\lim_{x\to a} \frac{f(x)}{g(x)} = \lim_{x\to a} \frac{f'(x)}{g'(x)} \quad \text{for } \frac{0}{0} \text{ or } \frac{\infty}{\infty}$$

**Limit of a Sequence for SGD Convergence:**
$$\lim_{t \to \infty} \|\theta_t - \theta^*\| = 0 \quad \text{with probability 1}$$

## Properties

1. **Uniqueness:** If $\lim_{x\to a} f(x)$ exists, it is unique. A function cannot approach two different values.

2. **Existence iff One-Sided Limits Agree:** $\lim_{x\to a} f(x)$ exists iff $\lim_{x\to a^-} f(x) = \lim_{x\to a^+} f(x)$.

3. **Locality:** Limits depend only on values near $a$, not at $a$ itself. Changing $f(a)$ does not affect $\lim_{x\to a} f(x)$.

4. **Linearity:** Limits distribute over addition and scalar multiplication: $\lim (af + bg) = a\lim f + b\lim g$.

5. **Preserves Order:** If $f(x) \leq g(x)$ near $a$, then $\lim_{x\to a} f(x) \leq \lim_{x\to a} g(x)$, provided both limits exist.

6. **Product and Quotient Preservation:** Limits distribute over products and quotients (when denominator limit is non-zero).

7. **Squeeze Property:** If a function is trapped between two functions with the same limit, it must share that limit.

8. **Infinite Limits:** $\lim_{x\to a} f(x) = \infty$ means the function grows without bound near $a$ (vertical asymptote).

9. **Limits at Infinity:** $\lim_{x\to\infty} f(x) = L$ means the function approaches $L$ as $x$ grows (horizontal asymptote).

10. **Sequential Characterisation:** $\lim_{x\to a} f(x) = L$ if and only if for every sequence $\{x_n\}$ with $x_n \to a$ and $x_n \neq a$, we have $f(x_n) \to L$.

## Step-by-Step Worked Examples

### Example 1: Direct Substitution and Factoring

Compute $\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$.

**Step 1:** Try direct substitution. $x = 2$ gives $\frac{4 - 4}{2 - 2} = \frac{0}{0}$, an indeterminate form.

**Step 2:** Factor the numerator.
$$\frac{x^2 - 4}{x - 2} = \frac{(x - 2)(x + 2)}{x - 2}$$

**Step 3:** Cancel the common factor $(x - 2)$, valid for $x \neq 2$.
$$\frac{(x - 2)(x + 2)}{x - 2} = x + 2$$

**Step 4:** Now substitute $x = 2$ into the simplified expression.
$$\lim_{x \to 2} (x + 2) = 2 + 2 = 4$$

**Answer:** $\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = 4$.

**Verification:** For $x = 1.9$, $\frac{3.61 - 4}{1.9 - 2} = \frac{-0.39}{-0.1} = 3.9$. For $x = 2.1$, $\frac{4.41 - 4}{2.1 - 2} = \frac{0.41}{0.1} = 4.1$. The values approach 4.

### Example 2: Squeeze Theorem

Compute $\lim_{x \to 0} x^2 \sin(1/x)$.

**Step 1:** Observe that $x^2 \to 0$ but $\sin(1/x)$ oscillates wildly as $x \to 0$ (it has no limit). We cannot use the product rule directly.

**Step 2:** Bound $\sin(1/x)$ using its range. Since $-1 \leq \sin(1/x) \leq 1$ for all $x \neq 0$, multiply by $x^2$ (non-negative for all $x$):
$$-x^2 \leq x^2 \sin(1/x) \leq x^2$$

**Step 3:** Apply the squeeze theorem. $\lim_{x\to 0} (-x^2) = 0$ and $\lim_{x\to 0} (x^2) = 0$.

Since $g(x) = -x^2 \leq f(x) \leq h(x) = x^2$, and both outer functions approach 0, the middle function must also approach 0.

**Answer:** $\lim_{x \to 0} x^2 \sin(1/x) = 0$.

### Example 3: L'Hôpital's Rule

Compute $\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$.

**Step 1:** Try direct substitution. $x = 0$ gives $\frac{1 - 1 - 0}{0} = \frac{0}{0}$, an indeterminate form.

**Step 2:** Apply L'Hôpital's rule. Differentiate numerator and denominator separately.
$$\lim_{x \to 0} \frac{e^x - 1 - x}{x^2} = \lim_{x \to 0} \frac{\frac{d}{dx}(e^x - 1 - x)}{\frac{d}{dx}(x^2)} = \lim_{x \to 0} \frac{e^x - 1}{2x}$$

**Step 3:** The result is still $0/0$ at $x = 0$. Apply L'Hôpital's rule again.
$$\lim_{x \to 0} \frac{e^x - 1}{2x} = \lim_{x \to 0} \frac{\frac{d}{dx}(e^x - 1)}{\frac{d}{dx}(2x)} = \lim_{x \to 0} \frac{e^x}{2}$$

**Step 4:** Now substitute $x = 0$.
$$\frac{e^0}{2} = \frac{1}{2}$$

**Answer:** $\lim_{x \to 0} \frac{e^x - 1 - x}{x^2} = \frac{1}{2}$.

### Example 4: Limit at Infinity with Exponential Dominance

Compute $\lim_{x \to \infty} \frac{x^3}{e^x}$.

**Step 1:** As $x \to \infty$, both numerator and denominator go to $\infty$, giving $\infty/\infty$.

**Step 2:** Apply L'Hôpital's rule.
$$\lim_{x \to \infty} \frac{x^3}{e^x} = \lim_{x \to \infty} \frac{3x^2}{e^x}$$

**Step 3:** Still $\infty/\infty$. Apply again.
$$\lim_{x \to \infty} \frac{3x^2}{e^x} = \lim_{x \to \infty} \frac{6x}{e^x}$$

**Step 4:** Still $\infty/\infty$. Apply a third time.
$$\lim_{x \to \infty} \frac{6x}{e^x} = \lim_{x \to \infty} \frac{6}{e^x}$$

**Step 5:** Now $\frac{6}{e^x} \to 0$ as $x \to \infty$.

**Answer:** $\lim_{x \to \infty} \frac{x^3}{e^x} = 0$.

This demonstrates that **exponentials dominate polynomials**: $e^x$ grows faster than any polynomial $x^p$ as $x \to \infty$.

### Example 5: Indeterminate Form $1^\infty$ — Continuous Compounding

Compute $\lim_{n \to \infty} \left(1 + \frac{0.05}{n}\right)^{n}$.

**Step 1:** This is the formula for continuously compounded interest at 5% annual rate.

**Step 2:** Let $y = \left(1 + \frac{0.05}{n}\right)^{n}$. Take natural logs.
$$\ln y = n \ln\left(1 + \frac{0.05}{n}\right) = \frac{\ln(1 + 0.05/n)}{1/n}$$

**Step 3:** As $n \to \infty$, $1/n \to 0$, so we have $0/0$ form. Let $h = 1/n$:
$$\lim_{n \to \infty} \ln y = \lim_{h \to 0} \frac{\ln(1 + 0.05h)}{h}$$

**Step 4:** Apply L'Hôpital's rule.
$$\lim_{h \to 0} \frac{\ln(1 + 0.05h)}{h} = \lim_{h \to 0} \frac{0.05/(1 + 0.05h)}{1} = 0.05$$

**Step 5:** Therefore $\ln y \to 0.05$, so $y \to e^{0.05}$.

**Answer:** $\lim_{n \to \infty} (1 + 0.05/n)^n = e^{0.05} \approx 1.05127$.

## Visual Interpretation

**Graphical Meaning of a Limit:**
```
f(x)
  |
L |----------o-----
  |         /
  |        /
  |-------/
  |      /|
  |     / |
  |    /  |
  |   /   |
  |  /    |
  | /     |
  |/      |
--+-------+------- x
  |       a
```
The point $(a, L)$ is shown as an open circle because the function need not be defined at $x = a$. The curve approaches the $y$-value $L$ as $x$ approaches $a$ from either side.

**One-Sided Limits:**
```
f(x)
  |
  |   -----o
  |  /    |
  | /      \
  |/        \
--+---a----+--- x
            \
             \
              ----o
```
Left-hand limit $\lim_{x\to a^-} f(x)$ approaches from below the $x$-axis; right-hand limit $\lim_{x\to a^+} f(x)$ approaches from above. They are unequal, so the two-sided limit does not exist — this is a jump discontinuity.

**Limits at Infinity:**
```
f(x)
  |
L1|-------o-------
  |        \
  |         \
--+----------\---- x
  |           \
  |            o-------L2
```
As $x \to -\infty$, $f(x) \to L_1$ (horizontal asymptote). As $x \to \infty$, $f(x) \to L_2$ (another horizontal asymptote).

**The Squeeze Theorem Visually:**
```
f(x)
  |    h(x) = x^2
  |   /\
  |  /  \
  | / f(x)\
  |/       \
--+---------> x
  |\       /
  | \     /
  |  \   /
  |   \ /
  |    v
  |    g(x) = -x^2
```
Between $x = -1$ and $x = 1$, the parabola $x^2$ bounds $x^2\sin(1/x)$ from above, and $-x^2$ bounds it from below. As both bounds squeeze to 0 at $x = 0$, the middle function must also approach 0.

## Common Mistakes

1. **Assuming the limit exists only if $f(a)$ is defined.** The limit depends only on values near $a$, not at $a$. A function can have a limit at $a$ even if it is undefined at $a$. For example, $\lim_{x\to 1} \frac{x^2-1}{x-1} = 2$ exists even though $f(1)$ is undefined.

2. **Cancelling factors without considering the domain.** In rational functions, factors like $(x-2)$ can be cancelled only when $x \neq 2$. The cancellation is valid for computing the limit because the limit ignores the point $x = 2$ itself.

3. **Applying L'Hôpital's rule to forms other than $0/0$ or $\infty/\infty$.** L'Hôpital's rule does **not** apply to $0/\infty$, $1/0$, or other forms. For example, $\lim_{x\to 0} \frac{\sin x}{x^2}$ is not $0/0$ — it is $0/0$, so it is actually applicable in this case, but $\lim_{x\to 0} \frac{1}{\sin x}$ is $1/0$ (not indeterminate) and L'Hôpital cannot be used. Always verify the form first.

4. **Forgetting to check that the limit of $g'(x)$ exists.** L'Hôpital's rule requires that $\lim f'(x)/g'(x)$ exists. If this limit does not exist, the original limit may still exist. For example, $\lim_{x\to\infty} \frac{x + \sin x}{x} = 1$, but applying L'Hôpital gives $\lim_{x\to\infty} \frac{1 + \cos x}{1}$, which does not exist.

5. **Treating $\infty$ as a number.** Limit expressions like $\lim_{x\to a} f(x) = \infty$ mean the function grows without bound, not that it equals some "infinite number." Algebraic manipulations with $\infty$ (e.g., $\infty - \infty = 0$) are invalid.

6. **Confusing limits with function values.** $\lim_{x\to a} f(x) = L$ does not imply $f(a) = L$. A function may approach $L$ near $a$ but be defined differently at $a$.

7. **Assuming $\lim_{x\to a} f(x)/g(x) = \lim f(x) / \lim g(x)$ when $\lim g(x) = 0$.** The quotient rule for limits requires the denominator limit to be non-zero.

8. **Misapplying the squeeze theorem with bounds that do not have the same limit.** Both bounding functions must have the same limit for the squeeze theorem to work.

9. **Thinking $\lim_{x\to 0} \frac{\sin(kx)}{x} = 1$.** Actually, $\lim_{x\to 0} \frac{\sin(kx)}{x} = k$. This follows from substituting $u = kx$: $\lim_{x\to 0} \frac{\sin(kx)}{x} = \lim_{u\to 0} \frac{\sin(u)}{u/k} = k \cdot \lim_{u\to 0} \frac{\sin u}{u} = k$.

10. **Applying L'Hôpital's rule when it's simpler to factor.** For $\lim_{x\to 1} \frac{x^3-1}{x-1}$, L'Hôpital gives $\lim_{x\to 1} \frac{3x^2}{1} = 3$, but factoring $(x-1)(x^2+x+1)/(x-1) = x^2+x+1$ yields the same answer more transparently.

## Interview Questions

### Beginner

1. **What does $\lim_{x \to a} f(x) = L$ mean?**
   *Answer: It means that as $x$ gets arbitrarily close to $a$ (from either side, but not equal to $a$), the function value $f(x)$ gets arbitrarily close to $L$. Formally, for any $\varepsilon > 0$, there exists $\delta > 0$ such that $0 < |x-a| < \delta$ implies $|f(x) - L| < \varepsilon$.*

2. **What is the difference between $\lim_{x\to a} f(x)$ and $f(a)$?**
   *Answer: The limit describes the behaviour of $f(x)$ as $x$ approaches $a$, regardless of what happens at $x = a$. The function value $f(a)$ is whatever the function outputs at exactly $x = a$. They may be equal (continuity) or different. The limit might exist even if $f(a)$ is undefined.*

3. **Evaluate $\lim_{x \to 3} (2x + 1)$.**
   *Answer: By direct substitution, $2(3) + 1 = 7$. Since $2x+1$ is a polynomial, it is continuous everywhere, so the limit equals the function value.*

4. **What is $\lim_{x \to 0} \frac{\sin x}{x}$? Why is this limit important?**
   *Answer: $\lim_{x \to 0} \frac{\sin x}{x} = 1$. This limit is essential for differentiating trigonometric functions — the derivative of $\sin x$ depends on it. It is proved using the squeeze theorem with geometric bounds on the unit circle.*

5. **What does it mean for a limit to not exist? Give an example.**
   *Answer: A limit does not exist when the function does not approach a single finite value as $x$ approaches $a$. Examples: (1) Jump discontinuity: $\lim_{x\to 0} \frac{|x|}{x}$ gives $-1$ from left, $+1$ from right. (2) Oscillatory behaviour: $\lim_{x\to 0} \sin(1/x)$ oscillates infinitely. (3) Unbounded growth: $\lim_{x\to 0} 1/x^2 = \infty$ (infinite limit).*

### Intermediate

1. **Explain L'Hôpital's rule and when it applies.**
   *Answer: L'Hôpital's rule states that for limits of the form $0/0$ or $\infty/\infty$, $\lim_{x\to a} \frac{f(x)}{g(x)} = \lim_{x\to a} \frac{f'(x)}{g'(x)}$, provided the limit of the ratio of derivatives exists (or is $\pm\infty$). It can be applied repeatedly if the indeterminate form persists. However, it requires: (1) the original limit is $0/0$ or $\infty/\infty$, (2) $f$ and $g$ are differentiable near $a$, and (3) the derivative limit exists.*

2. **How does the concept of limits relate to the vanishing gradient problem in deep learning?**
   *Answer: Training a deep network with $L$ layers involves gradient products: $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \prod_{i=2}^L W_i \prod_{i=1}^L \sigma'(z_i)$. As $L \to \infty$, if each term has magnitude $< 1$, the product $\to 0$ (vanishing). If $> 1$, the product $\to \infty$ (exploding). Analysing $\lim_{L\to\infty} \prod_{i=1}^L a_i$ explains why careful weight initialisation (e.g., Xavier initialisation keeping variance $\approx 1$) and normalisation (batch norm, layer norm) are critical.*

3. **Compute $\lim_{x \to 0} \frac{\tan x}{x}$.**
   *Answer: $\lim_{x \to 0} \frac{\tan x}{x} = \lim_{x \to 0} \frac{\sin x}{x \cos x} = \lim_{x \to 0} \frac{\sin x}{x} \cdot \lim_{x \to 0} \frac{1}{\cos x} = 1 \cdot \frac{1}{1} = 1$.*

4. **What is $\lim_{n\to\infty} (1 + r/n)^{nt}$ and why is it important in finance and ML?**
   *Answer: $\lim_{n\to\infty} (1 + r/n)^{nt} = e^{rt}$. In finance, this gives continuously compounded interest. In ML, this limit defines the exponential function $e^x$, which appears in: (1) softmax $p_i \propto e^{z_i}$, (2) cross-entropy loss $-\log p$, (3) Gaussian distributions $\propto e^{-x^2/2}$, (4) learning rate schedules like exponential decay $\alpha_t = \alpha_0 e^{-\beta t}$, and (5) attention mechanisms using exponentiated dot products.*

5. **Show that $\lim_{h \to 0} \frac{e^h - 1}{h} = 1$.**
   *Answer: Using the definition of $e = \lim_{n\to\infty} (1+1/n)^n$, let $n = 1/h$ (so $h \to 0$ corresponds to $n \to \infty$). Then $e^h - 1 = (1+1/n)^{1/n} - 1$. We can use the series expansion $e^h = 1 + h + h^2/2! + \cdots$, so $(e^h - 1)/h = 1 + h/2! + h^2/3! + \cdots \to 1$ as $h \to 0$. Alternatively, L'Hôpital's rule directly gives $\lim_{h\to 0} e^h/1 = 1$.*

### Advanced

1. **Prove $\lim_{x\to 0} \frac{\sin x}{x} = 1$ using the squeeze theorem with geometric arguments.**
   *Answer: Consider the unit circle with angle $x$ (in radians, $0 < x < \pi/2$). Draw the sector and two right triangles. The area of the small triangle $\triangle OAB$ is $\frac{1}{2}\sin x$. The area of the sector $OAB$ is $\frac{1}{2}x$. The area of the large triangle $\triangle OAC$ is $\frac{1}{2}\tan x$. Therefore: $\frac{1}{2}\sin x \leq \frac{1}{2}x \leq \frac{1}{2}\tan x$. Dividing by $\frac{1}{2}\sin x > 0$: $1 \leq \frac{x}{\sin x} \leq \frac{1}{\cos x}$. Taking reciprocals (reversing inequalities): $\cos x \leq \frac{\sin x}{x} \leq 1$. As $x \to 0$, $\cos x \to 1$. By the squeeze theorem, $\lim_{x\to 0} \frac{\sin x}{x} = 1$.*

2. **Derive the conditions on learning rates for SGD convergence (Robbins-Monro conditions) and explain their connection to limits.**
   *Answer: The Robbins-Monro conditions for almost-sure convergence of SGD are: $\sum_{t=1}^\infty \alpha_t = \infty$ and $\sum_{t=1}^\infty \alpha_t^2 < \infty$. The first condition $\sum \alpha_t = \infty$ ensures the algorithm can reach any point in parameter space regardless of initialisation (the steps are large enough in total). The second $\sum \alpha_t^2 < \infty$ ensures that the noise from stochastic gradients is damped sufficiently for convergence. A common choice $\alpha_t = 1/t$ satisfies both: $\sum 1/t = \infty$ (harmonic series diverges) and $\sum 1/t^2 = \pi^2/6 < \infty$ (converges). Analysis of infinite sums as limits of partial sums is essential here: $\lim_{T\to\infty} \sum_{t=1}^T \alpha_t = \infty$ and $\lim_{T\to\infty} \sum_{t=1}^T \alpha_t^2 < \infty$.*

3. **Analyse the numerical stability of computing $\log(1 + x)$ for small $x$ using limits. Why do libraries provide $\text{log1p}(x)$?**
   *Answer: For small $x$, computing $\log(1+x)$ directly suffers from catastrophic cancellation: $1 + x$ rounds to $1$ when $|x| < \varepsilon_{\text{machine}}$ (about $10^{-16}$ for double precision), losing all information about $x$. Using limits, $\lim_{x\to 0} \frac{\log(1+x)}{x} = 1$, so $\log(1+x) \approx x$ for small $x$. The $\text{log1p}(x)$ function uses the Taylor series $\log(1+x) = x - x^2/2 + x^3/3 - \cdots$, which is accurate for all $|x| < 1$. More precisely, $\text{log1p}(x)$ computes the limit accurately by avoiding the subtraction $1+x$ entirely. Similarly, $\text{expm1}(x) = e^x - 1$ avoids cancellation for small $x$ using $\lim_{x\to 0} (e^x - 1)/x = 1$. These functions are critical for numerically stable ML implementations, especially in loss functions with log-probabilities.*

## Practice Problems

### Easy

1. Compute $\lim_{x \to 4} (3x - 5)$.
2. Compute $\lim_{x \to -1} \frac{x^2 - 1}{x + 1}$.
3. Compute $\lim_{x \to 0} \frac{\sin(3x)}{x}$.
4. Evaluate $\lim_{x \to \infty} \frac{5x + 2}{3x - 1}$.
5. Determine $\lim_{x \to 2^+} \frac{1}{x - 2}$.

### Medium

1. Compute $\lim_{x \to 0} \frac{\sqrt{x + 1} - 1}{x}$.
2. Evaluate $\lim_{x \to 0} \frac{\sin(2x)}{\tan(3x)}$.
3. Compute $\lim_{x \to 3} \left(\frac{1}{x - 3} - \frac{5}{x^2 - x - 6}\right)$.
4. Find $\lim_{n \to \infty} \left(\frac{n}{n + 1}\right)^n$.
5. Use L'Hôpital's rule to compute $\lim_{x \to 0} \frac{e^{2x} - 1}{\sin x}$.

### Hard

1. Compute $\lim_{x \to 0} \frac{\sin(\sin x) - x}{x^3}$.
2. Evaluate $\lim_{x \to 0^+} x^x$.
3. Prove that $\lim_{n\to\infty} \left(1 + \frac{x}{n}\right)^n = e^x$ for any real $x$, using the definition of the limit and the exponential function.

## Solutions

### Easy Solutions

**1.** $\lim_{x \to 4} (3x - 5) = 3(4) - 5 = 7$. Direct substitution works because polynomials are continuous.

**2.** $\frac{x^2 - 1}{x + 1} = \frac{(x - 1)(x + 1)}{x + 1} = x - 1$ for $x \neq -1$. Thus $\lim_{x \to -1} (x - 1) = -2$.

**3.** $\lim_{x \to 0} \frac{\sin(3x)}{x} = 3 \cdot \lim_{x \to 0} \frac{\sin(3x)}{3x} = 3 \cdot 1 = 3$.

**4.** $\lim_{x \to \infty} \frac{5x + 2}{3x - 1} = \lim_{x \to \infty} \frac{5 + 2/x}{3 - 1/x} = \frac{5}{3}$.

**5.** As $x \to 2^+$, $x - 2 \to 0^+$ (small positive), so $\frac{1}{x-2} \to +\infty$.

### Medium Solutions

**1.** Rationalise: $\frac{\sqrt{x+1} - 1}{x} \cdot \frac{\sqrt{x+1} + 1}{\sqrt{x+1} + 1} = \frac{x}{x(\sqrt{x+1} + 1)} = \frac{1}{\sqrt{x+1} + 1}$. Then $\lim_{x\to 0} \frac{1}{\sqrt{x+1} + 1} = \frac{1}{1 + 1} = \frac{1}{2}$.

**2.** $\lim_{x\to 0} \frac{\sin(2x)}{\tan(3x)} = \lim_{x\to 0} \frac{\sin(2x)}{\sin(3x)/\cos(3x)} = \lim_{x\to 0} \frac{\sin(2x) \cos(3x)}{\sin(3x)}$. Using $\frac{\sin(2x)}{2x} \to 1$ and $\frac{\sin(3x)}{3x} \to 1$: $= \frac{2x \cdot \cos(3x)}{3x} \to \frac{2}{3} \cdot 1 = \frac{2}{3}$.

**3.** Factor $x^2 - x - 6 = (x - 3)(x + 2)$. Then $\frac{1}{x-3} - \frac{5}{(x-3)(x+2)} = \frac{(x+2) - 5}{(x-3)(x+2)} = \frac{x - 3}{(x-3)(x+2)} = \frac{1}{x+2}$ for $x \neq 3$. The limit is $\frac{1}{5}$.

**4.** $\lim_{n\to\infty} \left(\frac{n}{n+1}\right)^n = \lim_{n\to\infty} \left(1 - \frac{1}{n+1}\right)^n$. Let $m = n+1$: $\lim_{m\to\infty} \left(1 - \frac{1}{m}\right)^{m-1} = \lim_{m\to\infty} \frac{(1 - 1/m)^m}{1 - 1/m} = \frac{e^{-1}}{1} = \frac{1}{e} \approx 0.3679$.

**5.** Apply L'Hôpital: $\lim_{x\to 0} \frac{e^{2x} - 1}{\sin x} = \lim_{x\to 0} \frac{2e^{2x}}{\cos x} = \frac{2}{1} = 2$.

### Hard Solutions

**1.** Apply L'Hôpital's rule three times. First application: $\lim_{x\to 0} \frac{\cos(\sin x) \cdot \cos x - 1}{3x^2}$. Second application: $\lim_{x\to 0} \frac{-\sin(\sin x) \cdot \cos^2 x + \cos(\sin x) \cdot (-\sin x)}{6x}$. Third application: evaluate at $x = 0$: $\frac{0 + 0 + 0 + 0 + 0 + \cdots}{6}$. Alternatively, use Taylor series: $\sin(\sin x) = \sin(x - x^3/6 + \cdots) = (x - x^3/6) - (x - x^3/6)^3/6 + \cdots = x - x^3/3 + \cdots$. Then $\frac{\sin(\sin x) - x}{x^3} = \frac{-x^3/3 + \cdots}{x^3} \to -\frac{1}{3}$.

**2.** Let $y = x^x$, so $\ln y = x \ln x$. Then $\lim_{x\to 0^+} x \ln x = \lim_{x\to 0^+} \frac{\ln x}{1/x}$. Apply L'Hôpital: $\lim_{x\to 0^+} \frac{1/x}{-1/x^2} = \lim_{x\to 0^+} (-x) = 0$. So $\ln y \to 0$, hence $y \to e^0 = 1$. Thus $\lim_{x\to 0^+} x^x = 1$.

**3.** Let $y_n = (1 + x/n)^n$. Take $\ln y_n = n \ln(1 + x/n) = \frac{\ln(1 + x/n)}{1/n}$. As $n \to \infty$, $1/n \to 0$, so let $h = 1/n$: $\lim_{h\to 0} \frac{\ln(1 + xh)}{h}$. Apply L'Hôpital: $\lim_{h\to 0} \frac{x/(1 + xh)}{1} = x$. Thus $\ln y_n \to x$, so $y_n \to e^x$. This holds for any real $x$.

## Related Concepts

- **Continuity** (MATH-054) — A function is continuous at $a$ if $\lim_{x\to a} f(x) = f(a)$. Limits are the foundation of continuity.
- **Derivative** (MATH-055) — The derivative is defined as a limit of difference quotients.
- **Integral** — The definite integral is defined as a limit of Riemann sums.
- **Exponential Function** (MATH-050) — $e = \lim_{n\to\infty} (1 + 1/n)^n$ connects limits to exponentials.
- **Trigonometric Functions** (MATH-052) — The limit $\lim_{x\to 0} \sin x / x = 1$ is fundamental to calculus of trig functions.
- **Sequence and Series** — Convergence of sequences and series is analysed through limits of partial sums.

## Next Concepts

- **L'Hôpital's Rule (Advanced)** — More sophisticated applications including repeated applications and special cases.
- **Multivariable Limits** — Limits of functions $f: \mathbb{R}^n \to \mathbb{R}$, which require path independence.
- **Limit Superior and Inferior** — $\limsup$ and $\liminf$ for sequences that do not converge.
- **Asymptotic Analysis** — Big-O, little-o, and $\Theta$ notation for describing growth rates, essential for algorithm analysis.

## Summary

Limits describe the behaviour of a function as its input approaches a particular value. The formal $\varepsilon$-$\delta$ definition provides rigorous foundations for all of calculus. One-sided limits capture directional behaviour; their agreement is necessary and sufficient for the two-sided limit to exist. Limit laws allow algebraic manipulation of limits. Two special limits — $\lim_{x\to 0} \sin x / x = 1$ and $\lim_{n\to\infty} (1 + 1/n)^n = e$ — are foundational for trigonometric and exponential functions. L'Hôpital's rule resolves indeterminate forms by differentiating numerator and denominator. In AI/ML, limits underpin convergence analysis of optimisation algorithms, numerical stability considerations, the vanishing/exploding gradient problem, learning rate schedules, and the theoretical guarantees of universal approximation.

## Key Takeaways

- $\lim_{x\to a} f(x) = L$ means $f(x)$ gets arbitrarily close to $L$ as $x$ gets arbitrarily close to $a$.
- The $\varepsilon$-$\delta$ definition formalises "arbitrarily close" with inequalities.
- One-sided limits must agree for the two-sided limit to exist.
- Limit laws (sum, product, quotient, power, composition) enable algebraic limit computation.
- $\lim_{x\to 0} \frac{\sin x}{x} = 1$ and $\lim_{n\to\infty} (1 + 1/n)^n = e$ are two fundamental limits.
- L'Hôpital's rule resolves $0/0$ and $\infty/\infty$ indeterminate forms via differentiation.
- Limits define derivatives and integrals — they are the bedrock of calculus.
- In AI/ML, limits analyse convergence of gradient descent, numerical stability, vanishing/exploding gradients, and learning rate schedules.
- The squeeze theorem bounds a function between two others with the same limit.
- Exponentials dominate polynomials: $\lim_{x\to\infty} x^p / e^x = 0$ for any $p$.
