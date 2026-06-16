# Concept: Continuity

## Concept ID

MATH-054

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define continuity of a function at a point using the limit definition: $\lim_{x\to a} f(x) = f(a)$.
- Identify and classify discontinuities: removable, jump, infinite, and oscillatory.
- Apply the Intermediate Value Theorem to prove existence of roots and solutions.
- Understand continuity properties of common functions: polynomials, rationals, exponentials, logs, trig functions.
- Relate continuity to properties of activation functions (ReLU, sigmoid, tanh) and Lipschitz continuity in GANs.
- Analyse the role of continuous representations in neural networks, including implicit neural representations.

## Prerequisites

- Limits (MATH-053) — the definition of continuity is based on limits.
- Functions (MATH-044), domain (MATH-045), and range (MATH-046).
- Composite functions (MATH-047) and inverse functions (MATH-048).
- Basic familiarity with polynomial, exponential, logarithmic, and trigonometric functions.

## Definition

A function $f$ is **continuous at a point** $a$ in its domain if:

$$\lim_{x \to a} f(x) = f(a)$$

This definition implicitly requires three conditions:

1. **Existence of $f(a)$:** $a$ is in the domain of $f$.
2. **Existence of the limit:** $\lim_{x\to a} f(x)$ exists (finite).
3. **Equality:** The limit equals the function value.

If any of these conditions fails, $f$ is **discontinuous** at $a$.

**Continuity on an Interval:** $f$ is continuous on an open interval $(a, b)$ if it is continuous at every point in $(a, b)$. For a closed interval $[a, b]$, we require continuity on $(a, b)$ plus one-sided continuity at the endpoints: $\lim_{x\to a^+} f(x) = f(a)$ and $\lim_{x\to b^-} f(x) = f(b)$.

**Uniform Continuity:** A stronger notion: for every $\varepsilon > 0$, there exists a single $\delta > 0$ that works for all points in the domain (in contrast to pointwise continuity, where $\delta$ may depend on the point). Every continuous function on a closed bounded interval is uniformly continuous.

**Lipschitz Continuity:** $f$ is Lipschitz continuous with constant $K$ if $|f(x) - f(y)| \leq K|x - y|$ for all $x, y$ in the domain. This is stronger than (uniform) continuity, imposing a linear bound on how fast the function can change.

## Intuition

A function is continuous if you can draw its graph without lifting your pen. There are no holes, jumps, or breaks in the curve.

**Hole (Removable Discontinuity):** The graph has a missing point that can be filled in. Example: $f(x) = \frac{x^2-1}{x-1}$ has a hole at $x = 1$ because $f(1)$ is undefined, but $\lim_{x\to 1} f(x) = 2$.

**Jump (Step):** The graph has a sudden vertical leap. Example: $f(x) = \lfloor x \rfloor$ (floor function) jumps at every integer.

**Infinite (Vertical Asymptote):** The graph goes to $\pm\infty$. Example: $f(x) = 1/x$ at $x = 0$.

**Oscillatory:** The graph oscillates infinitely often near a point. Example: $f(x) = \sin(1/x)$ near $x = 0$ (if not defined as 0 at $x = 0$).

The epsilon-delta form of continuity: For every $\varepsilon > 0$, there exists $\delta > 0$ such that whenever $|x - a| < \delta$, we have $|f(x) - f(a)| < \varepsilon$. This says we can keep $f(x)$ within any desired tolerance $\varepsilon$ of $f(a)$ by restricting $x$ sufficiently close to $a$.

## Why This Concept Matters

Continuity is the mathematical foundation for modelling smooth, predictable phenomena. It guarantees that small changes in input produce small changes in output — a property we intuitively expect from well-behaved systems.

1. **Solving Equations:** The Intermediate Value Theorem (IVT) guarantees that a continuous function crossing a value must attain it. This proves the existence of solutions to equations without explicitly finding them.

2. **Optimisation:** Continuous functions on closed bounded intervals attain maximum and minimum values (Extreme Value Theorem), guaranteeing that optimisation problems have solutions.

3. **Neural Networks:** Activation functions must be carefully designed for continuity (e.g., ReLU is continuous but not differentiable at 0). Lipschitz continuity bounds how fast a network's output can change, which is critical for stability.

4. **Numerical Methods:** Newton's method, bisection, and other root-finding algorithms rely on continuity guarantees. The bisection method, for example, works because a continuous function that changes sign must cross zero (IVT).

5. **GAN Training:** Wasserstein GANs enforce Lipschitz continuity via gradient penalty to stabilise training. Without Lipschitz constraints, the discriminator can become too sharp, causing training instability.

## Historical Background

The concept of continuity evolved over centuries. Ancient Greek mathematicians implicitly used continuity in geometric reasoning, but a formal definition did not exist.

Augustin-Louis Cauchy (1789-1857) gave one of the first rigorous definitions of continuity in his *Cours d'Analyse* (1821). He defined a function $f$ as continuous at $a$ if the values of $f(x)$ approach $f(a)$ as $x$ approaches $a$.

Bernhard Riemann (1826-1866) further refined the concept and made important contributions to the theory of discontinuous functions and Fourier series. His work on trigonometric series forced mathematicians to grapple with functions that were discontinuous in surprising ways.

Karl Weierstrass (1815-1897) provided the modern $\varepsilon$-$\delta$ definition of continuity and shocked the mathematical world with his construction of a function that is continuous everywhere but differentiable nowhere — the Weierstrass function. This demonstrated that our geometric intuition about "smooth" curves is inadequate.

Henri Lebesgue (1875-1941) and others developed measure theory and integration, which allowed mathematicians to work with highly discontinuous functions. Today, continuity remains a fundamental concept, studied alongside weaker notions like semicontinuity and stronger notions like Lipschitz and H\"older continuity.

## Real World Examples

**Example 1: Temperature Throughout the Day.** Temperature as a function of time is continuous — it does not jump from one value to another instantaneously. This continuity allows weather prediction models to work with differential equations.

**Example 2: Volume of a Balloon as You Inflate It.** As you add air continuously, the volume increases continuously. There are no discrete jumps in volume. This relationship (pressure, volume, temperature) is modelled by continuous functions in thermodynamics.

**Example 3: Stock Price.** Stock prices are modelled as continuous stochastic processes (geometric Brownian motion) even though trades occur at discrete times. The continuous approximation enables the Black-Scholes option pricing model.

**Example 4: Elevator Position.** An elevator's height as a function of time is continuous — it cannot teleport between floors. However, its velocity is not continuous (it has a jerk at start/stop), illustrating different levels of smoothness.

**Example 5: Signal Strength.** Radio signal strength varies continuously with distance from the transmitter (inverse square law). This continuous relationship allows GPS trilateration and wireless positioning.

## AI/ML Relevance

Continuity is deeply embedded in the theory and practice of machine learning:

**1. Activation Functions and Continuity.** Most activation functions are designed to be continuous:
- **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$ is continuous and smooth (infinitely differentiable).
- **Tanh:** $\tanh(x)$ is continuous and smooth.
- **ReLU:** $\text{ReLU}(x) = \max(0, x)$ is continuous but not differentiable at $x = 0$. The left and right limits at 0 are both 0 (so it is continuous), but the left and right derivatives differ (0 vs 1), creating a "kink."
- **Leaky ReLU:** $f(x) = \max(\alpha x, x)$ for small $\alpha$ (e.g., 0.01) is also continuous with a kink at 0.
- **GELU (Gaussian Error Linear Unit):** $x \Phi(x)$ where $\Phi$ is the standard Gaussian CDF — smooth and continuous.

The continuity of activation functions ensures that small changes in the input (or weights) produce small changes in the output, which is essential for gradient-based training to work.

**2. Lipschitz Continuity in GAN Training.** The Wasserstein GAN (WGAN) formulation uses the Earth Mover (Wasserstein-1) distance, which requires the discriminator (critic) to be 1-Lipschitz continuous:
$$|D(x_1) - D(x_2)| \leq \|x_1 - x_2\|$$
This constraint is enforced through:
- Weight clipping (original WGAN): clamp weights to $[-c, c]$.
- Gradient penalty (WGAN-GP): add penalty term $(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2$.
- Spectral normalisation: normalise weights by the largest singular value.

Lipschitz continuity bounds how fast the discriminator's output can change, preventing mode collapse and improving training stability.

**3. Continuous Representations in Neural Fields.** Implicit neural representations (NeRF, SIREN, Fourier Features) model continuous functions $f: \mathbb{R}^n \to \mathbb{R}^m$ using neural networks. For example, NeRF represents a 3D scene as a continuous 5D function:
$$F_\Theta: (x, y, z, \theta, \phi) \to (R, G, B, \sigma)$$
The continuity of the network ensures that nearby points in 3D space produce similar colours and densities, enabling novel view synthesis and 3D reconstruction from 2D images.

**4. Universal Approximation Theorem.** The theorem states that a feedforward network with a single hidden layer and a continuous non-polynomial activation function can approximate any continuous function on a compact set arbitrarily well. This relies on the continuity of both the target function and the activation function. The density of neural network functions in the space of continuous functions is a fundamental result justifying the expressive power of neural networks.

**5. Loss Landscape Smoothness.** The continuity and differentiability of the loss function determine the behaviour of gradient descent. A Lipschitz-continuous gradient (L-smoothness) ensures:
$$\|\nabla L(\theta_1) - \nabla L(\theta_2)\| \leq L \|\theta_1 - \theta_2\|$$
This property guarantees that gradient descent with step size $\alpha < 2/L$ converges for convex functions. The smoothness constant $L$ determines the maximum stable learning rate.

**6. Robustness and Adversarial Examples.** Lipschitz continuity of a classifier provides robustness guarantees: if $|f(x) - f(y)| \leq K\|x - y\|$, then the classification cannot change unless the input changes by at least $1/K$ (for a binary classifier with margin 1). This connects continuity to adversarial robustness.

**7. Residual Networks and Continuous Flows.** ResNets can be viewed as discretisations of continuous ordinary differential equations:
$$h_{t+1} = h_t + f(h_t, \theta_t) \quad \text{vs.} \quad \frac{dh}{dt} = f(h(t), \theta(t))$$
Neural ODEs (Chen et al., 2018) treat the depth as a continuous variable, parameterising the derivative directly. The continuity of the transformation enables memory-efficient training (adjoint method) and natural handling of irregularly-sampled time series.

## Mathematical Explanation

**Algebra of Continuous Functions.** If $f$ and $g$ are continuous at $a$, then:
- $f + g$, $f - g$, and $f \cdot g$ are continuous at $a$.
- $f/g$ is continuous at $a$ provided $g(a) \neq 0$.
- $c \cdot f$ is continuous at $a$ for any constant $c$.
- $f \circ g$ (composition) is continuous at $a$ if $g$ is continuous at $a$ and $f$ is continuous at $g(a)$.
- $f^{-1}$ is continuous on its domain if $f$ is continuous and strictly monotonic.

**Continuity of Elementary Functions:**
- Polynomials: continuous on $\mathbb{R}$.
- Rational functions: continuous on their domain (all reals except where denominator = 0).
- $\sqrt[n]{x}$: continuous on $[0, \infty)$ for even $n$, on $\mathbb{R}$ for odd $n$.
- $\sin x$, $\cos x$: continuous on $\mathbb{R}$.
- $\tan x$, $\cot x$, $\sec x$, $\csc x$: continuous on their domains.
- $e^x$: continuous on $\mathbb{R}$.
- $\ln x$: continuous on $(0, \infty)$.

**Intermediate Value Theorem (IVT):** If $f$ is continuous on $[a, b]$ and $L$ is any number between $f(a)$ and $f(b)$ (inclusive), then there exists $c \in [a, b]$ such that $f(c) = L$.

**Extreme Value Theorem (EVT):** If $f$ is continuous on $[a, b]$, then $f$ attains both a maximum and a minimum value on $[a, b]$. That is, there exist $c, d \in [a, b]$ such that $f(c) \leq f(x) \leq f(d)$ for all $x \in [a, b]$.

**Classification of Discontinuities:**
1. **Removable:** $\lim_{x\to a} f(x)$ exists (finite) but $\neq f(a)$ (or $f(a)$ undefined). The "hole" can be filled by redefining $f(a)$.
2. **Jump:** $\lim_{x\to a^-} f(x)$ and $\lim_{x\to a^+} f(x)$ both exist but are not equal. The graph jumps from one value to another.
3. **Infinite:** $\lim_{x\to a} f(x) = \pm\infty$ (or one-sided limits are infinite). Vertical asymptote.
4. **Oscillatory:** The limit does not exist because the function oscillates infinitely near $a$.

**Lipschitz Continuity in Detail:**
$f$ is Lipschitz continuous on domain $D$ if there exists $K \geq 0$ such that for all $x, y \in D$:
$$|f(x) - f(y)| \leq K\|x - y\|$$
- The smallest such $K$ is the Lipschitz constant.
- If $f$ is differentiable with bounded derivative, then $K = \sup |f'(x)|$.
- Lipschitz continuity implies uniform continuity, which implies continuity.
- Examples: $f(x) = x$ has $K = 1$; $f(x) = \sin x$ has $K = 1$; $f(x) = x^2$ is not Lipschitz on $\mathbb{R}$ (derivative unbounded) but is Lipschitz on any bounded interval.

## Formula(s)

**Definition of Continuity:**
$$\lim_{x \to a} f(x) = f(a)$$

**$\varepsilon$-$\delta$ Definition:**
$$\forall \varepsilon > 0, \exists \delta > 0 \text{ such that } |x - a| < \delta \implies |f(x) - f(a)| < \varepsilon$$

**Intermediate Value Theorem:**
$$\text{If } f \in C([a,b]) \text{ and } f(a) \leq L \leq f(b) \text{ (or } f(b) \leq L \leq f(a)), \text{ then } \exists c \in [a,b] \text{ with } f(c) = L$$

**Extreme Value Theorem:**
$$\text{If } f \in C([a,b]), \text{ then } \exists c, d \in [a,b] \text{ such that } f(c) \leq f(x) \leq f(d) \text{ for all } x \in [a,b]$$

**Lipschitz Condition:**
$$|f(x) - f(y)| \leq K\|x - y\|$$

**WGAN-GP Gradient Penalty:**
$$\mathcal{L}_{\text{gp}} = \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} \left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]$$

## Properties

1. **Closure Under Operations:** Sums, differences, products, quotients (with non-zero denominator), and compositions of continuous functions are continuous.

2. **Intermediate Value Property:** Continuous functions map intervals to intervals. If $f$ is continuous, the image of any interval is also an interval (possibly a single point).

3. **Extreme Value Property:** Continuous functions on compact (closed and bounded) sets are bounded and attain their maximum and minimum.

4. **Removable Discontinuities:** If $\lim_{x\to a} f(x)$ exists but $f(a) \neq \lim f(x)$, the discontinuity can be removed by redefining $f(a)$.

5. **Monotone Functions:** A monotone function (increasing or decreasing) on an interval can only have jump discontinuities (no removables or oscillatory).

6. **Continuity of Inverse:** A continuous strictly monotonic function on an interval has a continuous inverse on its range.

7. **Uniform Continuity:** Every continuous function on a closed bounded interval is uniformly continuous — there exists a single $\delta$ that works for all points.

8. **Lipschitz vs. Differentiability:** Lipschitz continuity is intermediate between continuity and differentiability. A Lipschitz function is differentiable almost everywhere (Rademacher's theorem). A differentiable function with bounded derivative is Lipschitz.

9. **Baire Category:** The set of points where a function is discontinuous is an $F_\sigma$ set (countable union of closed sets). Conversely, any $F_\sigma$ set can be the discontinuity set of some function.

10. **Banach Fixed Point Theorem:** A contraction mapping (Lipschitz with $K < 1$) on a complete metric space has a unique fixed point, which can be found by iteration. This underlies the convergence proof of many iterative algorithms in ML.

## Step-by-Step Worked Examples

### Example 1: Checking Continuity at a Point

Determine if $f(x) = \begin{cases} \frac{x^2 - 4}{x - 2} & x \neq 2 \\ 4 & x = 2 \end{cases}$ is continuous at $x = 2$.

**Step 1:** Check if $f(2)$ is defined. $f(2) = 4$ by definition.

**Step 2:** Compute $\lim_{x\to 2} f(x)$. For $x \neq 2$, $f(x) = \frac{x^2 - 4}{x - 2} = \frac{(x-2)(x+2)}{x-2} = x+2$. So $\lim_{x\to 2} f(x) = \lim_{x\to 2} (x+2) = 4$.

**Step 3:** Check equality: $\lim_{x\to 2} f(x) = 4 = f(2)$.

**Answer:** $f$ is continuous at $x = 2$.

If instead $f(2) = 3$, we would have $\lim f(x) = 4 \neq 3 = f(2)$, giving a removable discontinuity.

### Example 2: Identifying and Classifying Discontinuities

Find and classify all discontinuities of $f(x) = \frac{x^2 - 1}{x^2 - 3x + 2}$.

**Step 1:** Factor numerator and denominator.
$$f(x) = \frac{(x-1)(x+1)}{(x-1)(x-2)}$$

**Step 2:** The domain excludes $x = 1$ and $x = 2$ (where denominator = 0).

**Step 3:** At $x = 1$: The factor $(x-1)$ cancels, giving $f(x) = \frac{x+1}{x-2}$ for $x \neq 1$. $\lim_{x\to 1} f(x) = \frac{1+1}{1-2} = -2$. The limit exists finitely, but $f(1)$ is undefined. This is a **removable discontinuity** (hole at $(1, -2)$).

**Step 4:** At $x = 2$: The denominator has an uncancelled factor $(x-2)$. $\lim_{x\to 2^-} f(x) = \lim_{x\to 2^-} \frac{x+1}{x-2} = \frac{3}{0^-} = -\infty$. $\lim_{x\to 2^+} f(x) = \frac{3}{0^+} = +\infty$. This is an **infinite discontinuity** (vertical asymptote).

**Answer:** $x = 1$ is a removable discontinuity; $x = 2$ is an infinite discontinuity.

### Example 3: Applying the Intermediate Value Theorem

Show that $f(x) = x^3 - 2x - 5$ has a root in the interval $[2, 3]$.

**Step 1:** Check that $f$ is continuous on $[2, 3]$. $f$ is a polynomial, hence continuous on $\mathbb{R}$.

**Step 2:** Compute $f(2)$ and $f(3)$.
$$f(2) = 8 - 4 - 5 = -1$$
$$f(3) = 27 - 6 - 5 = 16$$

**Step 3:** Since $f(2) = -1 < 0 < 16 = f(3)$, by the Intermediate Value Theorem, there exists $c \in (2, 3)$ such that $f(c) = 0$.

**Step 4 (Refinement):** Try $x = 2.5$: $f(2.5) = 15.625 - 5 - 5 = 5.625 > 0$. So the root is between 2 and 2.5. Try $x = 2.2$: $f(2.2) = 10.648 - 4.4 - 5 = 1.248 > 0$. Try $x = 2.1$: $f(2.1) = 9.261 - 4.2 - 5 = 0.061 > 0$. Try $x = 2.09$: $f(2.09) = 9.129 - 4.18 - 5 = -0.051 < 0$. So the root is approximately $c \approx 2.095$.

**Answer:** A root exists in $(2, 3)$ by IVT. Approximate root: $c \approx 2.095$.

### Example 4: Proving Lipschitz Continuity

Show that $f(x) = \sin(x^2)$ is Lipschitz continuous on $[0, 2]$ but not on $\mathbb{R}$.

**Step 1:** Compute the derivative: $f'(x) = 2x \cos(x^2)$.

**Step 2:** On $[0, 2]$, bound $|f'(x)| = |2x \cos(x^2)| \leq 2|x| \leq 4$. Since $f$ is differentiable with bounded derivative, $f$ is Lipschitz with $K = \sup_{[0,2]} |f'(x)| \leq 4$.

**Step 3:** On $\mathbb{R}$, as $x \to \infty$, $|f'(x)| = 2|x \cos(x^2)|$ is unbounded (since $\cos(x^2)$ equals $\pm 1$ infinitely often). Thus $f$ is not Lipschitz on $\mathbb{R}$.

**Answer:** $f$ is Lipschitz on $[0, 2]$ with $K \leq 4$ but not Lipschitz on $\mathbb{R}$.

### Example 5: Continuity of a Piecewise Function

Find $k$ such that $f(x) = \begin{cases} e^{x} & x \leq 0 \\ kx + 2 & x > 0 \end{cases}$ is continuous at $x = 0$.

**Step 1:** Compute $f(0)$: Since $0$ is in the first piece, $f(0) = e^0 = 1$.

**Step 2:** Compute $\lim_{x\to 0^-} f(x) = \lim_{x\to 0^-} e^x = e^0 = 1$.

**Step 3:** Compute $\lim_{x\to 0^+} f(x) = \lim_{x\to 0^+} (kx + 2) = k(0) + 2 = 2$.

**Step 4:** For continuity, the left and right limits must equal $f(0)$. The left limit already equals $f(0) = 1$. The right limit is $2$, which does not equal $1$ for any $k$.

**Answer:** No value of $k$ makes $f$ continuous at $x = 0$. The function always has a jump discontinuity at $0$ (jump from $1$ to $2$). If the second piece were $kx + 1$, then $k = 1$ would work.

## Visual Interpretation

**Continuity Visual:**
```
f(x)          f(x)          f(x)
  |             |             |
  |   ____       |   ____       |   ____
  |  /           |  /           |  /
  | /            | /       -----+-----/-------
--+-------> x   -+-------+-> x   | /
  |               |     /       |/
  |               |    /        |
Continuous     Removable     Jump
               Discontinuity Discontinuity
```

**Continuous:** The graph never breaks. For any $\varepsilon > 0$, we can find $\delta > 0$ so that all $x$-values within $\delta$ of $a$ produce $y$-values within $\varepsilon$ of $f(a)$.

**Removable Discontinuity:** There is a hole at $(a, L)$ — the limit exists but the function is either undefined or defined differently at $a$. Redefining the single point fills the hole.

**Jump Discontinuity:** The left and right limits exist but differ. The function jumps from one value to another.

**Infinite Discontinuity:** The function approaches $\pm\infty$ near $a$ — a vertical asymptote. The function cannot be made continuous at $a$ by any redefinition.

**Intermediate Value Theorem:**
```
f(x)
  |       ________
  |      / \
  |     /   \
  |    /     \
f(b) |-----\-----\---
  |   |      \    \
  |   |       \    \
L |---*--------*----\---
  |   |         \    \
  |   |          \    \
f(a) |------------\----\---
  |   |           \    \
  +---+---+---+---+----\--- x
      a   c1      c2   b
```
The horizontal line $y = L$ intersects the graph at at least one point $c \in [a, b]$. This is guaranteed by the IVT for any $L$ between $f(a)$ and $f(b)$.

## Common Mistakes

1. **Confusing continuity with differentiability.** A function can be continuous without being differentiable (e.g., $|x|$ at $x = 0$). Continuity is necessary but not sufficient for differentiability. The absolute value function has a "corner" at 0 — it is continuous but not differentiable there.

2. **Thinking the IVT guarantees existence of a root only when $f(a)$ and $f(b)$ have opposite signs.** While this is the common application, the IVT actually says that for *any* $L$ between $f(a)$ and $f(b)$, there exists $c$ with $f(c) = L$. So it also applies when both $f(a)$ and $f(b)$ are positive but some $L$ is between them.

3. **Assuming a function with a vertical asymptote is discontinuous at that point.** The function is not even defined there, so it is technically not "discontinuous" at that point (discontinuity requires the point to be in the domain). We say the function is discontinuous on its domain near the asymptote.

4. **Believing that piecewise-defined functions are necessarily discontinuous.** Many piecewise functions are continuous at the breakpoints if the pieces meet properly. For example, $f(x) = \begin{cases} x^2 & x \leq 1 \\ 2x - 1 & x > 1 \end{cases}$ is continuous at $x = 1$ since $\lim_{x\to 1^-} x^2 = 1$ and $\lim_{x\to 1^+} (2x-1) = 1$.

5. **Thinking the composition of continuous functions is always continuous.** The composition $f \circ g$ is continuous at $a$ if $g$ is continuous at $a$ and $f$ is continuous at $g(a)$. If $g(a)$ is a point of discontinuity of $f$, the composition may be discontinuous.

6. **Misapplying the EVT.** The Extreme Value Theorem requires a *closed bounded* interval and a *continuous* function. A continuous function on an open interval $(a, b)$ may not attain its maximum or minimum (e.g., $f(x) = x$ on $(0, 1)$ has no maximum).

7. **Equating Lipschitz continuity with having a bounded derivative.** They are equivalent for differentiable functions, but Lipschitz functions need not be differentiable everywhere (e.g., $|x|$ is Lipschitz with $K = 1$ but not differentiable at 0). Conversely, a function with unbounded derivative is not Lipschitz.

8. **Assuming ReLU is differentiable at 0.** ReLU is continuous at 0 (both sides approach 0) but not differentiable at 0. The left derivative is 0, the right derivative is 1. This is often finessed in practice by defining the derivative as 0, 1, or 0.5 at $x = 0$.

9. **Forgetting to check the domain when applying continuity of inverse functions.** The inverse of a continuous strictly monotonic function is continuous on its domain, but the domain of the inverse is the range of the original function, which may have different properties.

10. **Thinking that a function with a removable discontinuity is "almost continuous."** While the limit exists, the function is still technically discontinuous at that point. However, the function can be redefined at the point to make it continuous — this is called "removing" the discontinuity.

## Interview Questions

### Beginner

1. **What does it mean for a function $f$ to be continuous at $x = a$?**
   *Answer: A function is continuous at $x = a$ if $\lim_{x\to a} f(x) = f(a)$. This requires three things: (1) $f(a)$ exists, (2) $\lim_{x\to a} f(x)$ exists, and (3) these two values are equal. Graphically, you can draw the graph near $a$ without lifting your pen.*

2. **What are the different types of discontinuities?**
   *Answer: (1) Removable: the limit exists but does not equal the function value (or function undefined). (2) Jump: left and right limits exist but differ. (3) Infinite: the function approaches $\pm\infty$ near the point (vertical asymptote). (4) Oscillatory: the function oscillates infinitely near the point, so no limit exists.*

3. **State the Intermediate Value Theorem.**
   *Answer: If $f$ is continuous on $[a, b]$ and $L$ is any number between $f(a)$ and $f(b)$, then there exists $c \in [a, b]$ such that $f(c) = L$. In particular, if $f(a)$ and $f(b)$ have opposite signs, there is a root in $(a, b)$.*

4. **Is the function $f(x) = |x|$ continuous at $x = 0$?**
   *Answer: Yes. $f(0) = 0$, $\lim_{x\to 0} |x| = 0$, so they are equal. However, $|x|$ is not differentiable at $0$ — it has a corner. This shows continuity does not imply differentiability.*

5. **What is the Extreme Value Theorem?**
   *Answer: If $f$ is continuous on a closed bounded interval $[a, b]$, then $f$ attains both a maximum and minimum value on $[a, b]$. That is, there exist $c, d \in [a, b]$ such that $f(c) \leq f(x) \leq f(d)$ for all $x \in [a, b]$.*

### Intermediate

1. **Why is ReLU continuous but not differentiable at 0? How is this handled in practice during backpropagation?**
   *Answer: ReLU$(x) = \max(0, x)$. At $x = 0$, left limit = right limit = 0, so it is continuous. The left derivative is 0 ($\lim_{h\to0^-} \frac{0-0}{h} = 0$) and the right derivative is 1 ($\lim_{h\to0^+} \frac{h-0}{h} = 1$), so it is not differentiable at 0. In practice, during backpropagation, most frameworks (PyTorch, TensorFlow) define the derivative as 0 when $x \leq 0$ and 1 when $x > 0$. At exactly $x = 0$, they typically take 0 (though the choice is arbitrary since it occurs with probability zero).*

2. **Explain Lipschitz continuity and why it matters for Wasserstein GANs.**
   *Answer: A function $f$ is Lipschitz continuous with constant $K$ if $|f(x) - f(y)| \leq K\|x - y\|$ for all $x, y$. This means the function cannot change too quickly — its slope is bounded by $K$. In WGANs, the discriminator (critic) must be 1-Lipschitz to satisfy the Kantorovich-Rubinstein duality for the Wasserstein distance. This is enforced via gradient penalty (WGAN-GP) or spectral normalisation. Without Lipschitz constraints, the discriminator can assign arbitrarily different values to nearby points, causing training instability and mode collapse. Lipschitz continuity ensures the discriminator provides meaningful gradient information to the generator.*

3. **Prove that $\sin x$ is continuous on $\mathbb{R}$.**
   *Answer: We need to show $\lim_{x\to a} \sin x = \sin a$ for all $a$. Using the identity $\sin x - \sin a = 2 \cos(\frac{x+a}{2}) \sin(\frac{x-a}{2})$, we have $|\sin x - \sin a| = 2|\cos(\frac{x+a}{2})||\sin(\frac{x-a}{2})| \leq 2 \cdot 1 \cdot |\frac{x-a}{2}| = |x - a|$, since $|\sin \theta| \leq |\theta|$ and $|\cos| \leq 1$. Thus for any $\varepsilon > 0$, choose $\delta = \varepsilon$. If $|x - a| < \delta$, then $|\sin x - \sin a| < \varepsilon$. This proves continuity of $\sin$ on $\mathbb{R}$.*

4. **How does the IVT apply to the bisection method for root-finding?**
   *Answer: The bisection method starts with an interval $[a, b]$ where $f(a)$ and $f(b)$ have opposite signs. By the IVT, a root $c \in (a, b)$ exists. The method evaluates $f$ at the midpoint $m = (a+b)/2$, then selects the subinterval $[a, m]$ or $[m, b]$ where the signs are opposite. This halves the interval length each iteration, producing increasingly accurate approximations. After $n$ iterations, the error is bounded by $(b-a)/2^{n+1}$. The continuity of $f$ is essential — without it, the sign change does not guarantee a root.*

5. **What is the difference between continuity and uniform continuity? Give an example of a function that is continuous but not uniformly continuous.**
   *Answer: Continuity says: for each point $a$ and each $\varepsilon > 0$, there exists $\delta > 0$ (which may depend on $a$) such that $|x-a| < \delta$ implies $|f(x)-f(a)| < \varepsilon$. Uniform continuity says: for each $\varepsilon > 0$, there exists $\delta > 0$ (the same for all points) such that $|x-y| < \delta$ implies $|f(x)-f(y)| < \varepsilon$. Uniform continuity is stronger. Example: $f(x) = 1/x$ on $(0, 1)$ is continuous but not uniformly continuous. Near 0, the function changes too rapidly — no single $\delta$ works for all points simultaneously.*

### Advanced

1. **Prove the Extreme Value Theorem using the Bolzano-Weierstrass property.**
   *Answer: Let $f$ be continuous on $[a, b]$. First, show $f$ is bounded. Suppose not; then there exists a sequence $\{x_n\}$ with $|f(x_n)| \to \infty$. By Bolzano-Weierstrass, $\{x_n\}$ has a convergent subsequence $\{x_{n_k}\}$ with limit $x^* \in [a, b]$. Continuity of $f$ implies $f(x_{n_k}) \to f(x^*)$, contradicting $|f(x_{n_k})| \to \infty$. Thus $f$ is bounded. Let $M = \sup\{f(x) : x \in [a, b]\}$. By definition of supremum, there exists a sequence $\{x_n\}$ with $f(x_n) \to M$. Again by Bolzano-Weierstrass, some subsequence $\{x_{n_k}\}$ converges to $x^* \in [a, b]$. By continuity, $f(x_{n_k}) \to f(x^*)$. Since limits are unique, $f(x^*) = M$, so the maximum is attained. The proof for the minimum is analogous.*

2. **Explain how spectral normalisation enforces Lipschitz continuity in GAN discriminators. Derive the relationship between the Lipschitz constant and the largest singular value of each weight matrix.**
   *Answer: For a linear layer $h(x) = Wx + b$, the Lipschitz constant with respect to the $\ell_2$ norm is the spectral norm (largest singular value) of $W$: $\|h(x) - h(y)\| = \|W(x-y)\| \leq \|W\|_2 \|x-y\|$, where $\|W\|_2 = \sigma_{\max}(W) = \sqrt{\lambda_{\max}(W^T W)}$. For a deep network $f(x) = (g_L \circ \sigma \circ g_{L-1} \circ \cdots \circ \sigma \circ g_1)(x)$ where $g_i(x) = W_i x + b_i$ and $\sigma$ is a 1-Lipschitz activation (e.g., ReLU), the Lipschitz constant satisfies $\|f\|_{\text{Lip}} \leq \prod_{i=1}^L \|W_i\|_2$. Spectral normalisation constrains each $\|W_i\|_2 = 1$ by dividing each weight matrix by its spectral norm: $\overline{W}_i = W_i / \sigma_{\max}(W_i)$. This ensures the entire network is 1-Lipschitz, satisfying the WGAN constraint. The spectral norm is computed efficiently using power iteration during training.*

3. **Prove that the set of points where a monotone function is discontinuous is at most countable. Give an example of a monotone function with countably infinite discontinuities.**
   *Answer: Let $f$ be monotone increasing on $\mathbb{R}$. A monotone function can only have jump discontinuities (since one-sided limits always exist). For each point $a$ where $f$ is discontinuous, the jump $J(a) = \lim_{x\to a^+} f(x) - \lim_{x\to a^-} f(x) > 0$. For any $n \in \mathbb{N}$, the set $D_n = \{a : J(a) > 1/n\}$ is finite on any bounded interval (because the sum of jumps on a bounded interval cannot exceed the total increase). The full set of discontinuities $D = \bigcup_{n=1}^\infty D_n$ is a countable union of finite sets, hence at most countable. **Example:** The Cantor function (devil's staircase) is monotone increasing and continuous almost everywhere, but it is constant on intervals that are removed in constructing the Cantor set. A simpler example: define $f(x) = \sum_{n=1}^\infty \frac{\lfloor 2^n x \rfloor \mod 2}{2^n}$ (Thomae's function-like behaviour). More concretely: $f(x) = \sum_{n: q_n \leq x} 2^{-n}$ where $\{q_n\}$ is an enumeration of the rationals. This function jumps by $2^{-n}$ at each rational $q_n$, giving countably infinite discontinuities.*

## Practice Problems

### Easy

1. Find all points where $f(x) = \frac{x+2}{x^2 - 4}$ is discontinuous. Classify each.
2. Determine if $f(x) = \begin{cases} 2x + 1 & x \leq 3 \\ 10 - x & x > 3 \end{cases}$ is continuous at $x = 3$.
3. Use the IVT to show $f(x) = x^5 + x - 1$ has a root in $(0, 1)$.
4. Find $k$ so that $f(x) = \begin{cases} kx^2 & x \leq 2 \\ 4x - 4 & x > 2 \end{cases}$ is continuous at $x = 2$.
5. Is $f(x) = \frac{x^3 - 8}{x - 2}$ continuous at $x = 2$? If not, classify the discontinuity.

### Medium

1. Prove that $f(x) = \cos(1/x)$ has an oscillatory discontinuity at $x = 0$.
2. Show that the equation $x^3 + x - 5 = 0$ has exactly one real root.
3. Determine if $f(x) = \begin{cases} x \sin(1/x) & x \neq 0 \\ 0 & x = 0 \end{cases}$ is continuous at $x = 0$.
4. Find the Lipschitz constant of $f(x) = e^{-x^2}$ on $\mathbb{R}$.
5. Prove that if $f$ is continuous on $[0, 1]$ and $f(0) = f(1)$, then there exists $c \in [0, 1]$ such that $f(c) = f(c + 1/2)$.

### Hard

1. Prove that $f(x) = \begin{cases} \sin(1/x) & x \neq 0 \\ 0 & x = 0 \end{cases}$ is not continuous at $x = 0$, but $g(x) = \begin{cases} x \sin(1/x) & x \neq 0 \\ 0 & x = 0 \end{cases}$ is continuous at $x = 0$.
2. Construct a function that is continuous on $\mathbb{R}$ but differentiable nowhere (describe the Weierstrass function and its properties).
3. Prove that any Lipschitz function $f: \mathbb{R} \to \mathbb{R}$ is differentiable almost everywhere (Rademacher's theorem — outline the proof idea).

## Solutions

### Easy Solutions

**1.** Factor denominator: $x^2 - 4 = (x-2)(x+2)$. Domain excludes $x = \pm 2$. At $x = -2$: $\frac{x+2}{(x-2)(x+2)} = \frac{1}{x-2}$, so $\lim_{x\to -2} f(x) = -\frac{1}{4}$, a finite limit. This is removable (hole). At $x = 2$: $\lim_{x\to 2} \frac{1}{x-2} = \infty$, an infinite discontinuity (vertical asymptote).

**2.** $f(3) = 2(3) + 1 = 7$. $\lim_{x\to 3^-} f(x) = \lim_{x\to 3^-} (2x+1) = 7$. $\lim_{x\to 3^+} f(x) = \lim_{x\to 3^+} (10 - x) = 7$. Since left limit = right limit = $f(3) = 7$, $f$ is continuous at $x = 3$.

**3.** $f(0) = -1 < 0$, $f(1) = 1 + 1 - 1 = 1 > 0$. $f$ is a polynomial, hence continuous on $[0, 1]$. By IVT, there exists $c \in (0, 1)$ such that $f(c) = 0$.

**4.** $f(2) = k(2)^2 = 4k$. $\lim_{x\to 2^+} (4x - 4) = 4$. For continuity, $4k = 4 \implies k = 1$.

**5.** $f(2)$ is undefined (division by zero). Factor: $\frac{x^3 - 8}{x - 2} = \frac{(x-2)(x^2 + 2x + 4)}{x-2} = x^2 + 2x + 4$ for $x \neq 2$. $\lim_{x\to 2} f(x) = 4 + 4 + 4 = 12$. This is a removable discontinuity (hole at $(2, 12)$).

### Medium Solutions

**1.** For $x \neq 0$, as $x \to 0$, $\frac{1}{x}$ oscillates between $-\infty$ and $\infty$, so $\cos(1/x)$ oscillates between $-1$ and $1$ infinitely often. No single limit exists. Even defining $f(0) = 0$, the function oscillates too wildly to approach any value. This is an oscillatory discontinuity.

**2.** $f(x) = x^3 + x - 5$. $f'(x) = 3x^2 + 1 > 0$ for all $x$, so $f$ is strictly increasing. $f(1) = -3 < 0$, $f(2) = 5 > 0$. By IVT, there is at least one root in $(1, 2)$. Since $f$ is strictly monotonic (one-to-one), there is at most one root. Therefore exactly one real root exists.

**3.** Need to check $\lim_{x\to 0} x \sin(1/x)$. Since $-1 \leq \sin(1/x) \leq 1$, we have $-|x| \leq x \sin(1/x) \leq |x|$. By the squeeze theorem, $\lim_{x\to 0} x \sin(1/x) = 0$. Since $f(0) = 0$, we have $\lim_{x\to 0} f(x) = f(0)$. Therefore $f$ is continuous at $x = 0$.

**4.** $f'(x) = -2x e^{-x^2}$. On $\mathbb{R}$, $|f'(x)| = 2|x| e^{-x^2}$. This attains its maximum at $x = \pm 1/\sqrt{2}$: $|f'(1/\sqrt{2})| = \sqrt{2} e^{-1/2} = \sqrt{2/e} \approx 0.8578$. Since $f$ is differentiable and the derivative is bounded by $K = \sqrt{2/e}$, $f$ is Lipschitz with constant $\sqrt{2/e} \approx 0.858$.

**5.** Define $g(x) = f(x) - f(x + 1/2)$ on $[0, 1/2]$. $g$ is continuous (difference of continuous functions). $g(0) = f(0) - f(1/2)$. $g(1/2) = f(1/2) - f(1) = f(1/2) - f(0) = -g(0)$. If $g(0) = 0$, then $c = 0$ works. If $g(0) \neq 0$, then $g(0)$ and $g(1/2)$ have opposite signs. By IVT, there exists $c \in (0, 1/2)$ such that $g(c) = 0$, i.e., $f(c) = f(c+1/2)$.

### Hard Solutions

**1.** For $f(x) = \sin(1/x)$: Consider sequences $x_n = 1/(2\pi n)$ and $y_n = 1/(2\pi n + \pi/2)$. Then $f(x_n) = \sin(2\pi n) = 0$ and $f(y_n) = \sin(2\pi n + \pi/2) = 1$. Both sequences approach 0, but $f(x_n) \to 0$ while $f(y_n) \to 1$. Thus $\lim_{x\to 0} \sin(1/x)$ does not exist, so $f$ is not continuous at 0 regardless of how it is defined there. For $g(x) = x \sin(1/x)$: As shown in Medium #3, $|g(x)| \leq |x|$, so $\lim_{x\to 0} g(x) = 0 = g(0)$. Thus $g$ is continuous at 0. The factor $x$ "damps" the oscillations, forcing the function to 0 at the limit.

**2.** The Weierstrass function is defined by $W(x) = \sum_{n=0}^\infty a^n \cos(b^n \pi x)$, where $0 < a < 1$, $b$ is an odd integer, and $ab > 1 + 3\pi/2$ (typically $a = 1/2$, $b = 7$). Properties: (1) The series converges uniformly, so $W$ is continuous on $\mathbb{R}$. (2) Despite being continuous, $W$ is nowhere differentiable. The intuition: each term adds oscillations at a higher frequency ($b^n$), and the amplitudes $a^n$ do not decay fast enough to smooth out the derivatives. The fractal nature of $W$ means it has infinite "wiggliness" at every point. This was a revolutionary discovery (1872) that demolished the intuition that continuous functions must be differentiable except at isolated points.

**3.** (Proof Sketch) Rademacher's theorem states that a Lipschitz function $f: \mathbb{R}^n \to \mathbb{R}$ is differentiable almost everywhere (with respect to Lebesgue measure). Proof outline: (1) For each direction $v$, the directional derivative $D_v f(x) = \lim_{h\to 0} (f(x+hv) - f(x))/h$ exists almost everywhere by the differentiability almost everywhere of absolutely continuous functions on lines (a 1D result). (2) Since $f$ is Lipschitz, each $D_v f$ is measurable and Lipschitz in $v$. (3) One can show that for almost every $x$, the map $v \mapsto D_v f(x)$ is linear in $v$, so the gradient $\nabla f(x)$ exists. (4) The set of points where differentiability fails has measure zero. This result is important in ML for justifying that stochastic gradients exist almost surely even for non-smooth loss functions (e.g., hinge loss, ReLU networks).

## Related Concepts

- **Limits** (MATH-053) — Continuity is defined via limits; understanding limits is prerequisite.
- **Derivative** (MATH-055) — Differentiability implies continuity but not vice versa.
- **Partial Derivative** (MATH-056) — Continuity of partial derivatives implies differentiability.
- **Gradient** (MATH-058) — Lipschitz continuity of gradients ensures convergence of gradient descent.
- **Function** (MATH-044) — Continuity is a property of functions.
- **Polynomial Function** (MATH-049) — All polynomial functions are continuous.
- **Exponential Function** (MATH-050) — All exponential functions are continuous.

## Next Concepts

- **Differentiability** — A function is differentiable if its derivative exists; this is stronger than continuity.
- **Smoothness** — $C^k$ functions (continuous $k$-th derivative) and $C^\infty$ (infinitely differentiable).
- **Absolute Continuity** — A stronger form of continuity that is equivalent to being an indefinite integral.
- **H\"older Continuity** — $|f(x) - f(y)| \leq C|x-y|^\alpha$, intermediate between continuity and Lipschitz ($\alpha = 1$).

## Summary

Continuity formalises the intuitive notion that a function has no "breaks" or "jumps." A function $f$ is continuous at $a$ if $\lim_{x\to a} f(x) = f(a)$. Discontinuities are classified as removable, jump, infinite, or oscillatory. The Intermediate Value Theorem and Extreme Value Theorem are powerful consequences of continuity on closed intervals. In AI/ML, continuity of activation functions enables gradient-based training, Lipschitz continuity stabilises GAN training, and continuous representations (implicit neural representations) model signals with high fidelity. The relationship between continuity and differentiability is fundamental: differentiability implies continuity, but continuity does not imply differentiability.

## Key Takeaways

- $f$ continuous at $a$ iff $\lim_{x\to a} f(x) = f(a)$ (three conditions: $f(a)$ defined, limit exists, equality holds).
- Types of discontinuities: removable (hole), jump (step), infinite (asymptote), oscillatory.
- The IVT guarantees existence of intermediate values: continuous functions map intervals to intervals.
- The EVT guarantees continuous functions on $[a, b]$ attain their max and min.
- Sums, products, quotients, compositions of continuous functions are continuous.
- Lipschitz continuity ($|f(x) - f(y)| \leq K\|x-y\|$) is stronger than continuity and critical for GAN stability.
- ReLU is continuous but not differentiable at 0 — a subtlety handled by subgradients in practice.
- Neural ODEs treat depth as continuous, enabling memory-efficient training.
- The Universal Approximation Theorem relies on continuity of activation and target functions.
- Lipschitz continuity of gradients ($L$-smoothness) guarantees convergence of gradient descent.
