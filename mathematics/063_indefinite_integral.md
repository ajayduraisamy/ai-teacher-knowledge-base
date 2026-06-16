# Concept: Indefinite Integral

## Concept ID

MATH-063

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the indefinite integral as the family of antiderivatives of a function
- Understand the role of the constant of integration $C$
- Apply core integration techniques: substitution, integration by parts, partial fractions
- Compute antiderivatives for a wide range of elementary functions
- Recognise the relationship between indefinite integration and continuous-time models in machine learning

## Prerequisites

- Differential calculus (derivatives of elementary functions)
- Basic algebraic manipulation
- The Fundamental Theorem of Calculus
- Familiarity with the definite integral (MATH-062)

## Definition

The **indefinite integral** (or antiderivative) of a function $f(x)$ with respect to $x$ is defined as the set of all functions $F(x)$ such that $F'(x) = f(x)$. It is denoted:

$$
\int f(x) \, dx = F(x) + C
$$

where $C$ is an arbitrary real constant called the **constant of integration**.

Every differentiable function $f$ has infinitely many antiderivatives that differ only by an additive constant. If $F(x)$ is one antiderivative, then $F(x) + C$ for any constant $C$ is also an antiderivative, since the derivative of a constant is zero.

## Intuition

If differentiation measures the instantaneous rate of change of a function, indefinite integration reverses this process: given a rate of change $f(x)$, find the function $F(x)$ that has $f$ as its derivative.

Think of it as a "reverse engineering" of derivatives. If you know the slope at every point, indefinite integration reconstructs the original function, albeit with an unknown vertical offset $C$. This offset represents missing information about the function's absolute value — we know how it changes but not where it starts.

The notation $\int f(x) \, dx$ is due to Leibniz: the integral symbol $\int$ (elongated S for "sum") and $dx$ indicating the variable of integration. The indefinite integral represents the "general solution" to the differential equation $F'(x) = f(x)$.

## Why This Concept Matters

Indefinite integration is the primary tool for solving differential equations, which govern virtually all continuous processes in science and engineering. When we know how a system changes (through its derivative), the antiderivative tells us the system state.

In probability theory, the cumulative distribution function (CDF) is the indefinite integral of the probability density function (PDF). In physics, potential energy is the indefinite integral of force. In economics, total cost is the indefinite integral of marginal cost.

In machine learning, indefinite integrals appear in continuous-time models such as Neural ODEs, where the hidden state evolves according to a differential equation and must be "integrated up" from an initial condition. Understanding antiderivatives is essential for working with these models.

## Historical Background

The concept of the antiderivative was implicit in the work of **Isaac Barrow** (1630–1677), who recognised the inverse relationship between tangents and areas. **Isaac Newton** and **Gottfried Wilhelm Leibniz** independently formalised this relationship in the late 17th century, leading to the Fundamental Theorem of Calculus.

Leibniz developed the notation $\int$ in 1675, writing in a manuscript: "It will be useful to write $\int$ for _summa_, and $\int l$ for the sum of the $l$'s." His notation for the differential $dx$ and the integral $\int$ remains in use today, a testament to its clarity and utility.

The constant of integration $C$ was introduced by **Johann Bernoulli** (1667–1748) in the early 18th century. Bernoulli recognised that the antiderivative is not unique and that the additive constant accounts for this non-uniqueness.

The systematic development of integration techniques — substitution, integration by parts, partial fractions — was carried out by mathematicians throughout the 18th and 19th centuries, including Euler, Cauchy, and Riemann.

## Real World Examples

1. **Physics: Velocity to Position** — If an object's velocity is $v(t) = 9.8t$ m/s (free fall), its position is $s(t) = \int 9.8t \, dt = 4.9t^2 + C$. The constant $C$ is determined by the initial position $s(0) = s_0$.

2. **Economics: Marginal to Total Cost** — If marginal cost is $MC(q) = 3q^2 + 10$, then total cost is $TC(q) = \int (3q^2 + 10) \, dq = q^3 + 10q + C$, where $C$ is the fixed cost.

3. **Population Biology: Growth Rate to Population** — If a population grows at rate $r(t) = 500 e^{0.02t}$, the population at time $t$ is $P(t) = \int 500 e^{0.02t} \, dt = 25000 e^{0.02t} + C$.

4. **Engineering: Current to Charge** — The charge on a capacitor is $Q(t) = \int I(t) \, dt$, the indefinite integral of the current. The constant $C$ is the initial charge.

5. **Computer Graphics: Motion Blur** — Rendering motion blur requires integrating the light contribution over the time a shutter is open. The antiderivative of the time-varying radiance function determines the pixel colour.

## AI/ML Relevance

1. **Neural ODEs** — In Neural ODEs, the hidden state evolves as $\frac{d\mathbf{h}}{dt} = f(\mathbf{h}(t), t, \theta)$, and the output at time $t_1$ is $\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f(\mathbf{h}(t), t, \theta) \, dt$. The forward pass "solves" this integral equation. The adjoint sensitivity method computes gradients by solving another ODE backwards in time.

2. **Continuous Normalising Flows** — The instantaneous change-of-variables formula uses $\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\operatorname{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right)$, where $f$ is the vector field. The log-density at time $t_1$ is the integral of the trace over time: $\log p(\mathbf{z}(t_1)) = \log p(\mathbf{z}(t_0)) - \int_{t_0}^{t_1} \operatorname{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right) dt$.

3. **Cumulative Distribution Functions** — The CDF $F(x) = \int_{-\infty}^x f(t) \, dt$ is the indefinite integral (up to a constant) of the PDF. In Gaussian processes, the probit function $\Phi(z) = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} e^{-t^2/2} \, dt$ appears in classification.

4. **Potential Functions and Energy-Based Models** — Energy-based models define $p(\mathbf{x}) = \frac{e^{-E(\mathbf{x})}}{Z}$, where $Z$ is the normalisation constant. The energy function is related to the "potential" in a force field, a classic application of antiderivatives.

5. **Hamiltonian Neural Networks** — These models learn the Hamiltonian $H(\mathbf{q}, \mathbf{p})$ of a physical system. The equations of motion are $\dot{\mathbf{q}} = \frac{\partial H}{\partial \mathbf{p}}, \dot{\mathbf{p}} = -\frac{\partial H}{\partial \mathbf{q}}$, and integrating these ODEs requires solving indefinite integrals.

## Mathematical Explanation

### Basic Integration Rules

The indefinite integral of elementary functions follows directly from reversing known derivative formulas:

1. **Power Rule** (for $n \neq -1$):
   $$
   \int x^n \, dx = \frac{x^{n+1}}{n+1} + C
   $$

2. **Logarithmic Rule**:
   $$
   \int \frac{1}{x} \, dx = \ln|x| + C
   $$

3. **Exponential Rule**:
   $$
   \int e^x \, dx = e^x + C, \quad \int a^x \, dx = \frac{a^x}{\ln a} + C
   $$

4. **Trigonometric Rules**:
   $$
   \int \sin x \, dx = -\cos x + C, \quad \int \cos x \, dx = \sin x + C
   $$

   $$
   \int \sec^2 x \, dx = \tan x + C, \quad \int \csc^2 x \, dx = -\cot x + C
   $$

   $$
   \int \sec x \tan x \, dx = \sec x + C, \quad \int \csc x \cot x \, dx = -\csc x + C
   $$

5. **Inverse Trigonometric**:
   $$
   \int \frac{dx}{\sqrt{1 - x^2}} = \arcsin x + C, \quad \int \frac{dx}{1 + x^2} = \arctan x + C
   $$

### Linearity

Indefinite integration is linear:

$$
\int [af(x) + bg(x)] \, dx = a \int f(x) \, dx + b \int g(x) \, dx
$$

### Integration by Substitution

If $u = g(x)$ and $du = g'(x) \, dx$:

$$
\int f(g(x)) g'(x) \, dx = \int f(u) \, du
$$

This is the reverse of the chain rule.

### Integration by Parts

From the product rule $(uv)' = u'v + uv'$:

$$
\int u \, dv = uv - \int v \, du
$$

Choosing $u$ and $dv$ strategically is key. The LIATE rule (Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential) prioritises $u$ choices.

### Partial Fractions

For rational functions $\frac{P(x)}{Q(x)}$ with $\deg(P) < \deg(Q)$, factor $Q(x)$ into linear and irreducible quadratic factors, then decompose into simpler fractions.

### Trigonometric Integrals and Substitutions

For expressions involving $\sqrt{a^2 - x^2}$, use $x = a\sin\theta$; for $\sqrt{a^2 + x^2}$, use $x = a\tan\theta$; for $\sqrt{x^2 - a^2}$, use $x = a\sec\theta$.

## Formula(s)

1. **Indefinite Integral Definition**:
   $$
   \int f(x) \, dx = F(x) + C \iff F'(x) = f(x)
   $$

2. **Power Rule**:
   $$
   \int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)
   $$

3. **Log Rule**:
   $$
   \int \frac{1}{x} \, dx = \ln|x| + C
   $$

4. **Exponential**:
   $$
   \int e^{ax} \, dx = \frac{1}{a}e^{ax} + C
   $$

5. **Substitution**:
   $$
   \int f(g(x))g'(x) \, dx = \int f(u) \, du, \quad u = g(x)
   $$

6. **Integration by Parts**:
   $$
   \int u \, dv = uv - \int v \, du
   $$

## Properties

1. **Additive Constant**: The constant of integration $C$ can be any real number; all antiderivatives differ by a constant.

2. **Linearity**: $\int (af + bg) \, dx = a\int f \, dx + b\int g \, dx$.

3. **Inverse of Differentiation**: $\frac{d}{dx} \left(\int f(x) \, dx\right) = f(x)$, and $\int f'(x) \, dx = f(x) + C$.

4. **Non-Uniqueness**: The indefinite integral produces a family of functions, not a single function.

5. **Domain Specificity**: The antiderivative $\ln|x|$ ensures the integral of $1/x$ is defined on intervals not containing $0$, both for $x > 0$ and $x < 0$.

6. **Integration Does Not Preserve Elementary Functions**: The indefinite integral of an elementary function is not always elementary (e.g., $\int e^{-x^2} \, dx$).

## Step-by-Step Worked Examples

### Example 1: Basic Polynomials

**Problem**: Compute $\int (4x^3 - 6x^2 + 2x - 5) \, dx$.

**Solution**:

Step 1: Apply linearity.

$$
\int (4x^3 - 6x^2 + 2x - 5) \, dx = 4\int x^3 \, dx - 6\int x^2 \, dx + 2\int x \, dx - 5\int 1 \, dx
$$

Step 2: Apply the power rule to each term.

- $\int x^3 \, dx = \frac{x^4}{4} + C_1$
- $\int x^2 \, dx = \frac{x^3}{3} + C_2$
- $\int x \, dx = \frac{x^2}{2} + C_3$
- $\int 1 \, dx = x + C_4$

Step 3: Combine.

$$
4 \cdot \frac{x^4}{4} - 6 \cdot \frac{x^3}{3} + 2 \cdot \frac{x^2}{2} - 5x + C
$$

where $C = 4C_1 - 6C_2 + 2C_3 - 5C_4$.

Result:
$$
\int (4x^3 - 6x^2 + 2x - 5) \, dx = x^4 - 2x^3 + x^2 - 5x + C
$$

Verification: $\frac{d}{dx}(x^4 - 2x^3 + x^2 - 5x + C) = 4x^3 - 6x^2 + 2x - 5$. ✓

### Example 2: Integration by Substitution

**Problem**: Compute $\int \frac{\ln x}{x} \, dx$.

**Solution**:

Step 1: Identify the substitution. Let $u = \ln x$.

Step 2: Compute $du$. $du = \frac{1}{x} \, dx$.

Step 3: Rewrite the integral.

$$
\int \frac{\ln x}{x} \, dx = \int u \, du
$$

Step 4: Integrate.

$$
\int u \, du = \frac{u^2}{2} + C
$$

Step 5: Substitute back.

$$
\int \frac{\ln x}{x} \, dx = \frac{(\ln x)^2}{2} + C
$$

Verification: $\frac{d}{dx} \frac{(\ln x)^2}{2} = \frac{1}{2} \cdot 2 \ln x \cdot \frac{1}{x} = \frac{\ln x}{x}$. ✓

### Example 3: Integration by Parts

**Problem**: Compute $\int x^2 e^x \, dx$.

**Solution**:

Step 1: Apply integration by parts. Let $u = x^2$, $dv = e^x \, dx$, so $du = 2x \, dx$, $v = e^x$.

$$
\int x^2 e^x \, dx = x^2 e^x - \int 2x e^x \, dx
$$

Step 2: Apply integration by parts again to $\int x e^x \, dx$. Let $u = x$, $dv = e^x \, dx$, so $du = dx$, $v = e^x$.

$$
\int x e^x \, dx = x e^x - \int e^x \, dx = x e^x - e^x + C_1
$$

Step 3: Substitute back.

$$
\int x^2 e^x \, dx = x^2 e^x - 2(x e^x - e^x) + C = x^2 e^x - 2x e^x + 2e^x + C
$$

Result:
$$
\int x^2 e^x \, dx = e^x(x^2 - 2x + 2) + C
$$

Verification: $\frac{d}{dx} [e^x(x^2 - 2x + 2)] = e^x(x^2 - 2x + 2) + e^x(2x - 2) = e^x x^2$. ✓

### Example 4: Partial Fractions

**Problem**: Compute $\int \frac{x + 5}{x^2 + x - 2} \, dx$.

**Solution**:

Step 1: Factor the denominator.

$x^2 + x - 2 = (x+2)(x-1)$

Step 2: Decompose.

$$
\frac{x + 5}{(x+2)(x-1)} = \frac{A}{x+2} + \frac{B}{x-1}
$$

Multiply: $x + 5 = A(x-1) + B(x+2) = (A+B)x + (-A + 2B)$

Step 3: Solve for $A$ and $B$.

Coefficient of $x$: $A + B = 1$
Constant: $-A + 2B = 5$

Adding: $3B = 6 \implies B = 2$, then $A = -1$.

Step 4: Integrate.

$$
\int \frac{x + 5}{x^2 + x - 2} \, dx = \int \frac{-1}{x+2} \, dx + \int \frac{2}{x-1} \, dx
$$

$$
= -\ln|x+2| + 2\ln|x-1| + C = \ln\left|\frac{(x-1)^2}{x+2}\right| + C
$$

### Example 5: Trigonometric Substitution

**Problem**: Compute $\int \frac{dx}{\sqrt{4 - x^2}}$.

**Solution**:

Step 1: Use the substitution $x = 2\sin\theta$, so $dx = 2\cos\theta \, d\theta$.

Step 2: Rewrite.

$$
\int \frac{dx}{\sqrt{4 - x^2}} = \int \frac{2\cos\theta \, d\theta}{\sqrt{4 - 4\sin^2\theta}} = \int \frac{2\cos\theta \, d\theta}{2\sqrt{1 - \sin^2\theta}} = \int \frac{2\cos\theta \, d\theta}{2|\cos\theta|}
$$

For $-\pi/2 < \theta < \pi/2$, $\cos\theta > 0$, so $|\cos\theta| = \cos\theta$.

$$
= \int d\theta = \theta + C
$$

Step 3: Substitute back. Since $x = 2\sin\theta$, $\theta = \arcsin(x/2)$.

$$
\int \frac{dx}{\sqrt{4 - x^2}} = \arcsin\left(\frac{x}{2}\right) + C
$$

## Visual Interpretation

The indefinite integral represents a family of curves, all having the same derivative $f(x)$ at every point. Imagine a slope field: at each $x$, all curves in the family have the same tangent slope $f(x)$, but they are vertically shifted relative to each other by the constant $C$.

Choosing a specific value of $C$ corresponds to selecting one particular curve from this family, typically determined by an initial condition (e.g., $F(0) = 5$). This is why indefinite integrals appear in initial value problems: $\frac{dy}{dx} = f(x), y(x_0) = y_0$ has a unique solution $y(x) = \int f(x) \, dx$ with $C$ chosen to satisfy $y(x_0) = y_0$.

## Common Mistakes

1. **Forgetting $+C$**: The most common error. Every indefinite integral must include the constant of integration. Without $C$, the expression is incomplete.

2. **Power rule for $n = -1$**: $\int x^{-1} \, dx = \ln|x| + C$, not $\frac{x^0}{0} + C$. This is the single most important exception to the power rule.

3. **Incorrect back-substitution**: After substitution, failing to express the final answer back in terms of the original variable.

4. **Wrong $u$ and $dv$ in integration by parts**: Choosing $u$ as the more complicated function (instead of following LIATE) often makes the integral harder.

5. **Missing absolute values in logarithms**: $\int \frac{1}{x} \, dx = \ln|x| + C$, not $\ln x + C$. The absolute value ensures the antiderivative is defined for negative $x$ as well.

6. **Algebraic errors in partial fractions**: Forgetting to perform polynomial long division when $\deg(P) \geq \deg(Q)$, or incorrectly solving for the decomposition coefficients.

7. **Assuming every function has an elementary antiderivative**: Many functions (e.g., $e^{-x^2}$, $\frac{\sin x}{x}$, $\frac{1}{\ln x}$) do not have closed-form antiderivatives. This is known as Liouville's theorem in differential algebra.

8. **Confusing indefinite and definite integrals**: The indefinite integral is a family of functions $F(x) + C$; the definite integral is a number $\int_a^b f(x) \, dx$. They are related through the FTC.

## Interview Questions

### Beginner

1. **Q**: What is an indefinite integral?
   **A**: The indefinite integral $\int f(x) \, dx = F(x) + C$ is the set of all antiderivatives of $f$, where $F'(x) = f(x)$ and $C$ is an arbitrary constant.

2. **Q**: Compute $\int (3x^2 + 2x - 1) \, dx$.
   **A**: $x^3 + x^2 - x + C$.

3. **Q**: What is $\int \frac{1}{x} \, dx$ and why is the absolute value needed?
   **A**: $\ln|x| + C$. The absolute value ensures the antiderivative is defined for all $x \neq 0$, since $\ln x$ is only defined for $x > 0$ but $\frac{1}{x}$ is defined for $x \neq 0$.

4. **Q**: Compute $\int e^{3x} \, dx$.
   **A**: $\frac{1}{3}e^{3x} + C$.

5. **Q**: Why do we include $+C$ in indefinite integrals but not in definite integrals?
   **A**: The indefinite integral represents a family of antiderivatives differing by constants; the constant accounts for all possibilities. The definite integral evaluates to a specific number, and the $+C$ cancels out when computing $F(b) - F(a)$.

### Intermediate

1. **Q**: Use substitution to compute $\int \frac{2x}{x^2 + 1} \, dx$.
   **A**: Let $u = x^2 + 1$, $du = 2x \, dx$. Then $\int \frac{du}{u} = \ln|u| + C = \ln(x^2 + 1) + C$.

2. **Q**: Compute $\int x \cos x \, dx$ using integration by parts.
   **A**: Let $u = x$, $dv = \cos x \, dx$, so $du = dx$, $v = \sin x$. Then $\int x \cos x \, dx = x \sin x - \int \sin x \, dx = x \sin x + \cos x + C$.

3. **Q**: How does the constant of integration relate to initial conditions in differential equations?
   **A**: The general solution to $y' = f(x)$ is $y = \int f(x) \, dx = F(x) + C$. An initial condition $y(x_0) = y_0$ determines $C = y_0 - F(x_0)$, giving a unique particular solution.

4. **Q**: Compute $\int \tan x \, dx$.
   **A**: $\tan x = \frac{\sin x}{\cos x}$. Let $u = \cos x$, $du = -\sin x \, dx$. Then $\int \tan x \, dx = -\int \frac{du}{u} = -\ln|\cos x| + C = \ln|\sec x| + C$.

5. **Q**: What is the relationship between indefinite integration and Neural ODEs?
   **A**: In Neural ODEs, the hidden state is $\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f(\mathbf{h}(t), t, \theta) \, dt$, which is a definite integral. The vector field $f$ is the time derivative, so the hidden state at $t_1$ is an antiderivative evaluated from $t_0$ to $t_1$.

### Advanced

1. **Q**: Derive the formula for integration by parts from the product rule and explain its use in computing expectations in probabilistic models.
   **A**: $(uv)' = u'v + uv'$. Integrate: $uv = \int u'v + \int uv'$. Rearranged: $\int uv' = uv - \int u'v$. In probability, this is used for expectations like $\mathbb{E}[X] = \int x f(x) \, dx$ where $f$ is a PDF. Integration by parts connects $\mathbb{E}[X]$ to the CDF: $\mathbb{E}[X] = \int_0^{\infty} (1-F(x)) \, dx - \int_{-\infty}^0 F(x) \, dx$ for a continuous random variable with CDF $F$.

2. **Q**: Liouville's theorem in differential algebra states that most elementary functions do not have elementary antiderivatives. Give three examples relevant to ML and explain how they are handled in practice.
   **A**: (1) $\int e^{-x^2} \, dx$ (Gaussian CDF) — handled via the error function $\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2} \, dt$; (2) $\int \frac{e^x}{x} \, dx$ — handled via the exponential integral $\text{Ei}(x)$; (3) $\int \frac{\sin x}{x} \, dx$ (sinc function) — handled via the sine integral $\text{Si}(x)$. In ML, these special functions are typically implemented as numerical approximations or through series expansions.

3. **Q**: Explain how the adjoint sensitivity method for Neural ODEs uses the concept of indefinite integration to compute gradients without storing intermediate states. What is the computational cost?
   **A**: The adjoint method defines $\mathbf{a}(t) = \frac{\partial L}{\partial \mathbf{h}(t)}$ and derives $\frac{d\mathbf{a}}{dt} = -\mathbf{a}(t)^T \frac{\partial f}{\partial \mathbf{h}}$. To compute $\frac{\partial L}{\partial \theta}$, we need $\int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f}{\partial \theta} \, dt$. The key insight: instead of solving from $t_0$ forward (which would require storing all intermediate $\mathbf{h}(t)$), we solve the adjoint ODE backwards from $t_1$ to $t_0$, recomputing $\mathbf{h}(t)$ backwards as needed. The cost is $O(1)$ memory (no intermediate storage) and roughly $O(T)$ time per gradient computation (one forward ODE solve plus one backward adjoint solve), each requiring $O(N)$ operations per timestep where $N$ is the hidden dimension.

## Practice Problems

### Easy

1. Compute $\int (2x^5 - 4x^3 + x) \, dx$.

2. Compute $\int \frac{3}{x} \, dx$.

3. Compute $\int 5e^{2x} \, dx$.

4. Compute $\int (2\sin x + 3\cos x) \, dx$.

5. Compute $\int (2x + 1)^2 \, dx$ (expand first).

### Medium

1. Use substitution to compute $\int x e^{x^2} \, dx$.

2. Use integration by parts to compute $\int \ln x \, dx$.

3. Compute $\int \frac{dx}{x^2 + 4}$.

4. Compute $\int x \sec^2 x \, dx$ (integration by parts).

5. Use substitution to compute $\int \frac{e^x}{1 + e^x} \, dx$.

### Hard

1. Compute $\int e^{2x} \cos x \, dx$ (integration by parts twice).

2. Compute $\int \frac{2x^2 + 3x + 1}{(x-1)(x+1)^2} \, dx$ (partial fractions with a repeated factor).

3. Compute $\int \sqrt{1 - x^2} \, dx$ (trigonometric substitution).

## Solutions

### Easy Solutions

**Solution 1**: $\int (2x^5 - 4x^3 + x) \, dx = \frac{2x^6}{6} - \frac{4x^4}{4} + \frac{x^2}{2} + C = \frac{x^6}{3} - x^4 + \frac{x^2}{2} + C$.

**Solution 2**: $\int \frac{3}{x} \, dx = 3 \ln|x| + C$.

**Solution 3**: $\int 5 e^{2x} \, dx = \frac{5}{2} e^{2x} + C$.

**Solution 4**: $\int (2\sin x + 3\cos x) \, dx = -2\cos x + 3\sin x + C$.

**Solution 5**: $(2x+1)^2 = 4x^2 + 4x + 1$. $\int (4x^2 + 4x + 1) \, dx = \frac{4x^3}{3} + 2x^2 + x + C$.

### Medium Solutions

**Solution 1**: $u = x^2$, $du = 2x\,dx$, $x\,dx = du/2$. $\int x e^{x^2} \, dx = \frac{1}{2} \int e^u \, du = \frac{e^{x^2}}{2} + C$.

**Solution 2**: Let $u = \ln x$, $dv = dx$, so $du = \frac{1}{x}dx$, $v = x$.
$\int \ln x \, dx = x \ln x - \int x \cdot \frac{1}{x} \, dx = x \ln x - \int dx = x \ln x - x + C$.

**Solution 3**: $\int \frac{dx}{x^2 + 4} = \frac{1}{2} \arctan\left(\frac{x}{2}\right) + C$ (using $\int \frac{dx}{x^2 + a^2} = \frac{1}{a}\arctan\frac{x}{a} + C$).

**Solution 4**: Let $u = x$, $dv = \sec^2 x \, dx$, $du = dx$, $v = \tan x$.
$\int x \sec^2 x \, dx = x \tan x - \int \tan x \, dx = x \tan x + \ln|\cos x| + C = x \tan x - \ln|\sec x| + C$.

**Solution 5**: Let $u = 1 + e^x$, $du = e^x \, dx$.
$\int \frac{e^x}{1 + e^x} \, dx = \int \frac{du}{u} = \ln|u| + C = \ln(1 + e^x) + C$.

### Hard Solutions

**Solution 1**: Let $I = \int e^{2x} \cos x \, dx$.
Parts: $u = \cos x$, $dv = e^{2x} dx$, $du = -\sin x\,dx$, $v = \frac{1}{2}e^{2x}$.
$I = \frac{1}{2}e^{2x}\cos x + \frac{1}{2} \int e^{2x} \sin x \, dx$.

Now for $\int e^{2x} \sin x \, dx$: $u = \sin x$, $dv = e^{2x}dx$, $du = \cos x\,dx$, $v = \frac{1}{2}e^{2x}$.
$\int e^{2x} \sin x \, dx = \frac{1}{2}e^{2x}\sin x - \frac{1}{2} \int e^{2x} \cos x \, dx = \frac{1}{2}e^{2x}\sin x - \frac{1}{2}I$.

Substitute: $I = \frac{1}{2}e^{2x}\cos x + \frac{1}{2}\left(\frac{1}{2}e^{2x}\sin x - \frac{1}{2}I\right) = \frac{1}{2}e^{2x}\cos x + \frac{1}{4}e^{2x}\sin x - \frac{1}{4}I$.
Thus $\frac{5}{4}I = e^{2x}\left(\frac{1}{2}\cos x + \frac{1}{4}\sin x\right)$.
$I = \frac{4}{5}e^{2x}\left(\frac{1}{2}\cos x + \frac{1}{4}\sin x\right) + C = \frac{e^{2x}}{5}(2\cos x + \sin x) + C$.

**Solution 2**: Decompose: $\frac{2x^2+3x+1}{(x-1)(x+1)^2} = \frac{A}{x-1} + \frac{B}{x+1} + \frac{C}{(x+1)^2}$.
Multiply: $2x^2 + 3x + 1 = A(x+1)^2 + B(x-1)(x+1) + C(x-1)$.
$= A(x^2+2x+1) + B(x^2-1) + C(x-1) = (A+B)x^2 + (2A+C)x + (A - B - C)$.
Solving: $A+B=2$, $2A+C=3$, $A-B-C=1$. From first two: $B=2-A$, $C=3-2A$.
Sub into third: $A - (2-A) - (3-2A) = 1 \implies A - 2 + A - 3 + 2A = 1 \implies 4A - 5 = 1 \implies A = \frac{3}{2}$.
$B = \frac{1}{2}$, $C = 0$.
$\int = \frac{3}{2}\int \frac{dx}{x-1} + \frac{1}{2}\int \frac{dx}{x+1} = \frac{3}{2}\ln|x-1| + \frac{1}{2}\ln|x+1| + C$.

**Solution 3**: Let $x = \sin\theta$, $dx = \cos\theta\,d\theta$, $\sqrt{1-x^2} = \cos\theta$.
$\int \sqrt{1-x^2} \, dx = \int \cos\theta \cdot \cos\theta \, d\theta = \int \cos^2\theta \, d\theta = \int \frac{1+\cos 2\theta}{2} \, d\theta$.
$= \frac{1}{2}\theta + \frac{1}{4}\sin 2\theta + C = \frac{1}{2}\theta + \frac{1}{2}\sin\theta\cos\theta + C$.
Since $x = \sin\theta$, $\theta = \arcsin x$, $\cos\theta = \sqrt{1-x^2}$.
Result: $\int \sqrt{1-x^2} \, dx = \frac{1}{2}\arcsin x + \frac{x}{2}\sqrt{1-x^2} + C$.

## Related Concepts

- **Definite Integral**: The indefinite integral evaluated between limits (MATH-062)
- **Fundamental Theorem of Calculus**: Links indefinite and definite integrals
- **Differentiation**: The inverse operation of indefinite integration
- **Differential Equations**: Equations involving derivatives are solved by indefinite integration
- **Riemann Sum**: The sum definition that underlies the integral
- **Integration by Parts**: Technique derived from the product rule
- **Partial Fractions**: Technique for integrating rational functions

## Next Concepts

- **Multiple Integrals**: Indefinite integration in multiple variables (MATH-064)
- **Improper Integrals**: Integrals with infinite limits or singularities
- **Numerical Integration**: Approximating integrals without closed-form antiderivatives
- **Integral Transforms**: Laplace and Fourier transforms
- **Differential Equations**: Solving ODEs and PDEs using integration

## Summary

The indefinite integral $\int f(x) \, dx = F(x) + C$ is the family of all antiderivatives of $f$, where $F'(x) = f(x)$ and $C$ is the constant of integration. It is the inverse operation of differentiation, reconstructing a function from its derivative up to an additive constant.

Key techniques include substitution (reverse chain rule), integration by parts (reverse product rule), and partial fractions (for rational functions). These techniques enable the integration of a wide range of elementary functions.

In machine learning, indefinite integration appears in continuous-time models (Neural ODEs, continuous normalising flows), probability (CDFs, entropy integrals), and energy-based models. Understanding antiderivatives is essential for working with differential equations and continuous dynamics.

## Key Takeaways

- The indefinite integral reverses differentiation: $\int f' = f + C$
- $+C$ must always be included — antiderivatives are families, not single functions
- Substitution reverses the chain rule; integration by parts reverses the product rule
- $\int x^{-1} \, dx = \ln|x| + C$ is the exception to the power rule
- Many elementary functions lack elementary antiderivatives (Liouville's theorem)
- Indefinite integrals appear in Neural ODEs as continuous-time hidden state evolution
- The CDF is the indefinite integral of the PDF
- Integration by parts is essential for computing expectations in probability
- Partial fractions integrate rational functions by algebraic decomposition
- Trigonometric substitution handles integrals with square roots of quadratics
