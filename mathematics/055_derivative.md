# Concept: Derivative

## Concept ID

MATH-055

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the derivative $f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}$ and interpret it as the instantaneous rate of change and slope of the tangent line.
- Apply differentiation rules: power rule, product rule, quotient rule, chain rule, and derivatives of elementary functions.
- Compute derivatives of polynomials, exponentials, logarithms, trigonometric functions, and their compositions.
- Use derivatives to analyse function behaviour: monotonicity, concavity, local extrema, and inflection points.
- Connect derivatives to gradient descent, backpropagation, and optimisation in machine learning.
- Understand higher-order derivatives and their geometric meaning (acceleration, curvature).

## Prerequisites

- Limits (MATH-053) — the derivative is defined via a limit.
- Continuity (MATH-054) — differentiability implies continuity.
- Functions (MATH-044), composite functions (MATH-047).
- Basic algebra and exponent/logarithm rules.

## Definition

The **derivative** of a function $f$ at a point $x$ is defined as the limit of the difference quotient:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

provided this limit exists. If the limit exists, $f$ is said to be **differentiable** at $x$.

**Alternative notation:** $\frac{df}{dx}$, $\frac{d}{dx}f(x)$, $\dot{f}(x)$ (physics), $D_x f$.

**Left and Right Derivatives:**
$$f'_-(x) = \lim_{h \to 0^-} \frac{f(x+h) - f(x)}{h}, \quad f'_+(x) = \lim_{h \to 0^+} \frac{f(x+h) - f(x)}{h}$$

The derivative exists iff both one-sided derivatives exist and are equal.

**Geometric Interpretation:** $f'(a)$ is the slope of the tangent line to the graph of $f$ at the point $(a, f(a))$. The tangent line equation is $y - f(a) = f'(a)(x - a)$.

**Physical Interpretation:** If $s(t)$ is position at time $t$, then $s'(t)$ is instantaneous velocity and $s''(t)$ is acceleration.

## Intuition

Imagine driving a car. Your speedometer shows instantaneous velocity — how fast you are going at exactly this moment. The derivative is the mathematical version of the speedometer reading. It captures the **instantaneous rate of change**.

Approximate the derivative by secant lines: pick two points $(x, f(x))$ and $(x+h, f(x+h))$ and compute the slope $\frac{f(x+h)-f(x)}{h}$. As $h$ shrinks to 0, the secant line approaches the tangent line, and its slope approaches the derivative.

If $f'(x) > 0$, the function is increasing at $x$; if $f'(x) < 0$, it is decreasing; if $f'(x) = 0$, the function has a horizontal tangent (potential local extremum or saddle point).

The derivative measures **sensitivity**: how much does $f$ change when the input changes by a tiny amount? This is exactly what we need for optimisation — which direction should we move in parameter space to decrease the loss?

## Why This Concept Matters

The derivative is arguably the single most important concept in calculus for its applications to science, engineering, and machine learning:

1. **Optimisation:** Finding minima/maxima of functions is central to all of engineering and science. The derivative gives us the gradient, which tells us which direction to move to decrease a function.

2. **Gradient Descent:** Every neural network is trained using variants of gradient descent: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$. The gradient $\nabla L$ is a vector of partial derivatives. Without derivatives, there is no gradient descent and no deep learning.

3. **Backpropagation:** Backpropagation computes derivatives of the loss with respect to every parameter in a neural network using the chain rule of calculus.

4. **Physics and Engineering:** Velocity, acceleration, force (as derivative of potential energy), current (as derivative of charge), and countless other physical quantities are defined as derivatives.

5. **Economics:** Marginal cost, marginal revenue, and marginal utility are derivatives — they measure the effect of producing or consuming one more unit.

6. **Control Theory:** PID controllers use proportional, integral, and derivative terms to regulate systems.

## Historical Background

The concept of the derivative emerged in the 17th century from two problems: finding tangent lines to curves (geometric) and describing instantaneous motion (physical).

**Pierre de Fermat** (1601-1665) developed a method for finding tangents using "adequality" — a precursor to the derivative. He also developed a method for finding maxima and minima by setting the derivative to zero.

**Isaac Newton** (1642-1727) developed the "method of fluxions" (fluent = variable, fluxion = derivative). He used derivatives to formulate the laws of motion and universal gravitation. Newton's notation $\dot{f}$ (dot above the variable) for time derivatives is still used in physics.

**Gottfried Wilhelm Leibniz** (1646-1716) independently developed calculus using the notation $\frac{dy}{dx}$ that we use today. His notation emphasised the ratio of infinitesimals, making the chain rule $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$ intuitively natural.

The priority dispute between Newton and Leibniz divided European mathematics for decades. Britain followed Newton's notation and fell behind continental mathematics, while the continent followed Leibniz's superior notation.

In the 19th century, **Augustin-Louis Cauchy** and **Karl Weierstrass** put derivatives on a rigorous foundation using limits, removing the need for "infinitesimals" (which were not rigorously defined at the time).

## Real World Examples

**Example 1: Velocity and Acceleration.** A ball thrown upward has height $h(t) = -4.9t^2 + 20t + 1.5$ meters at time $t$ seconds. $h'(t) = -9.8t + 20$ m/s is velocity. At $t = 0$, $v = 20$ m/s upward. At $t = 20/9.8 \approx 2.04$ s, $v = 0$ (maximum height). $h''(t) = -9.8$ m/s$^2$ is acceleration due to gravity.

**Example 2: Marginal Cost.** A factory's cost to produce $x$ units is $C(x) = 5000 + 25x + 0.01x^2$ dollars. $C'(x) = 25 + 0.02x$ is the marginal cost — the cost of producing one more unit. At $x = 1000$, marginal cost is $25 + 20 = 45$ dollars per unit.

**Example 3: Population Growth.** If a population grows as $P(t) = P_0 e^{rt}$, then $P'(t) = rP_0 e^{rt} = rP(t)$. The growth rate is proportional to the current population, which characterises exponential growth.

**Example 4: Drug Concentration Decay.** The concentration of a drug in the bloodstream decays as $C(t) = C_0 e^{-kt}$. $C'(t) = -k C_0 e^{-kt} = -k C(t)$. The half-life is $t_{1/2} = \ln 2 / k$.

**Example 5: Temperature Change.** Newton's law of cooling: $T'(t) = -k(T(t) - T_{\text{env}})$. The rate of temperature change is proportional to the temperature difference with the environment.

## AI/ML Relevance

Derivatives are the engine of deep learning:

**1. Gradient Descent.** The fundamental optimisation algorithm for neural networks:
$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$
Here $\nabla L(\theta) = \left(\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots\right)$ is the gradient vector. Each component $\partial L / \partial \theta_i$ is a partial derivative.

**2. Backpropagation.** Backpropagation is the algorithm for computing $\nabla L(\theta)$ efficiently. For a simple 2-layer network:
$$h = \sigma(W_1 x + b_1)$$
$$\hat{y} = W_2 h + b_2$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

The derivative $\partial L / \partial W_2 = (\hat{y} - y) \cdot h^T$ is straightforward. For $W_1$, using the chain rule:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial (W_1 x + b_1)} \cdot \frac{\partial (W_1 x + b_1)}{\partial W_1}$$
$$= (\hat{y} - y) \cdot W_2^T \cdot \sigma'(W_1 x + b_1) \cdot x^T$$

**3. Vanishing and Exploding Gradients.** In deep networks, gradients are products of many derivatives. Consider a deep network with $L$ layers using sigmoid activation $\sigma(z)$ where $\sigma'(z) \leq 0.25$. Then:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \cdot \prod_{i=2}^L W_i \cdot \prod_{i=1}^L \sigma'(z_i)$$
As $L$ grows, if each $\sigma'(z_i) < 1$, the product tends to 0 (vanishing gradient). This explains why sigmoid/tanh activations make deep networks hard to train, motivating ReLU and careful initialisation schemes.

**4. Higher-Order Derivatives in Optimisation.** Newton's method uses the second derivative (Hessian) to accelerate convergence:
$$\theta_{t+1} = \theta_t - [\nabla^2 L(\theta_t)]^{-1} \nabla L(\theta_t)$$
Methods like L-BFGS approximate the Hessian. Adam optimiser uses estimates of first and second moments of gradients.

**5. Activation Function Derivatives.**
- Sigmoid: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, max value 0.25 (vanishing for large $|x|$).
- Tanh: $\tanh'(x) = 1 - \tanh^2(x)$, max value 1.
- ReLU: $\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ (no saturation for $x > 0$, but dead neurons for $x < 0$).
- GELU: smooth approximation of ReLU, derivative involves normal CDF and PDF.

**6. Neural Tangent Kernel (NTK).** The NTK $K(x, x') = \nabla_\theta f(x; \theta)^T \nabla_\theta f(x'; \theta)$ characterises how infinitesimal weight updates affect the network output. In the infinite-width limit, the NTK becomes deterministic, connecting neural network training to kernel regression.

**7. Physics-Informed Neural Networks (PINNs).** PINNs incorporate physical laws (PDEs) into the loss function using automatic differentiation:
$$L = \|f(x, t) - u(x, t)\|^2 + \left\|\frac{\partial u}{\partial t} - \mathcal{N}[u]\right\|^2$$

## Mathematical Explanation

**Differentiation Rules:**

1. **Constant Rule:** $\frac{d}{dx}[c] = 0$
2. **Power Rule:** $\frac{d}{dx}[x^n] = nx^{n-1}$ for any real $n$
3. **Constant Multiple Rule:** $\frac{d}{dx}[cf(x)] = cf'(x)$
4. **Sum/Difference Rule:** $\frac{d}{dx}[f(x) \pm g(x)] = f'(x) \pm g'(x)$
5. **Product Rule:** $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$
6. **Quotient Rule:** $\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$
7. **Chain Rule:** $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$

**Derivatives of Elementary Functions:**
- $\frac{d}{dx}[x^n] = nx^{n-1}$
- $\frac{d}{dx}[e^x] = e^x$
- $\frac{d}{dx}[a^x] = a^x \ln a$
- $\frac{d}{dx}[\ln x] = \frac{1}{x}$
- $\frac{d}{dx}[\sin x] = \cos x$
- $\frac{d}{dx}[\cos x] = -\sin x$
- $\frac{d}{dx}[\tan x] = \sec^2 x$
- $\frac{d}{dx}[\arcsin x] = \frac{1}{\sqrt{1-x^2}}$
- $\frac{d}{dx}[\arctan x] = \frac{1}{1+x^2}$

**Higher-Order Derivatives:**
- Second derivative: $f''(x) = \frac{d}{dx}[f'(x)]$ — measures concavity and acceleration.
- If $f''(x) > 0$, the graph is concave up (convex); if $f''(x) < 0$, concave down.
- Inflection points: where $f''(x) = 0$ and changes sign.

## Formula(s)

**Definition:**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Product Rule:**
$$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$$

**Quotient Rule:**
$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

**Chain Rule:**
$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

**Power Rule:**
$$\frac{d}{dx}[x^n] = nx^{n-1}$$

**Exponential:**
$$\frac{d}{dx}[e^x] = e^x, \quad \frac{d}{dx}[a^x] = a^x \ln a$$

**Logarithmic:**
$$\frac{d}{dx}[\ln x] = \frac{1}{x}, \quad \frac{d}{dx}[\log_a x] = \frac{1}{x \ln a}$$

**Trigonometric:**
$$\frac{d}{dx}[\sin x] = \cos x, \quad \frac{d}{dx}[\cos x] = -\sin x, \quad \frac{d}{dx}[\tan x] = \sec^2 x$$

## Properties

1. **Linearity:** $\frac{d}{dx}[af(x) + bg(x)] = af'(x) + bg'(x)$ — the derivative is a linear operator.

2. **Differentiability Implies Continuity:** If $f'(a)$ exists, then $f$ is continuous at $a$.

3. **Continuity Does Not Imply Differentiability:** $f(x) = |x|$ is continuous everywhere but not differentiable at $x = 0$ (sharp corner).

4. **Product Rule for n Functions:** $(f_1 f_2 \cdots f_n)' = f_1' f_2 \cdots f_n + f_1 f_2' \cdots f_n + \cdots + f_1 f_2 \cdots f_n'$.

5. **Leibniz Rule for Higher-Order Products:** $(f \cdot g)^{(n)} = \sum_{k=0}^n \binom{n}{k} f^{(k)} g^{(n-k)}$.

6. **Derivative of Inverse:** If $y = f(x)$ and $f$ is invertible, then $\frac{dy}{dx} = \frac{1}{dx/dy}$.

7. **Mean Value Theorem:** If $f$ is continuous on $[a, b]$ and differentiable on $(a, b)$, then there exists $c \in (a, b)$ such that $f'(c) = \frac{f(b) - f(a)}{b - a}$.

8. **Fermat's Theorem:** If $f$ has a local extremum at $c$ and $f'(c)$ exists, then $f'(c) = 0$.

9. **Derivative and Monotonicity:** If $f'(x) > 0$ on an interval, $f$ is increasing; if $f'(x) < 0$, $f$ is decreasing.

10. **Derivative and Concavity:** If $f''(x) > 0$, $f$ is concave up (convex); if $f''(x) < 0$, $f$ is concave down.

## Step-by-Step Worked Examples

### Example 1: Derivative from First Principles

Find the derivative of $f(x) = x^2 + 3x$ using the limit definition.

**Step 1:** Write the difference quotient.
$$\frac{f(x+h) - f(x)}{h} = \frac{(x+h)^2 + 3(x+h) - (x^2 + 3x)}{h}$$

**Step 2:** Expand the numerator.
$$= \frac{x^2 + 2xh + h^2 + 3x + 3h - x^2 - 3x}{h}$$

**Step 3:** Simplify.
$$= \frac{2xh + h^2 + 3h}{h} = 2x + h + 3$$

**Step 4:** Take the limit as $h \to 0$.
$$f'(x) = \lim_{h \to 0} (2x + h + 3) = 2x + 3$$

**Answer:** $f'(x) = 2x + 3$.

### Example 2: Product Rule

Find $f'(x)$ for $f(x) = x^2 e^x$.

**Step 1:** Identify $u = x^2$ and $v = e^x$.
**Step 2:** Compute $u' = 2x$ and $v' = e^x$.
**Step 3:** Apply the product rule.
$$f'(x) = 2x \cdot e^x + x^2 \cdot e^x = e^x (2x + x^2)$$

**Answer:** $f'(x) = e^x (x^2 + 2x)$.

### Example 3: Quotient Rule

Find $f'(x)$ for $f(x) = \frac{x^2 + 1}{x - 2}$.

**Step 1:** Identify $u = x^2 + 1$ and $v = x - 2$.
**Step 2:** Compute $u' = 2x$ and $v' = 1$.
**Step 3:** Apply the quotient rule.
$$f'(x) = \frac{(2x)(x-2) - (x^2+1)(1)}{(x-2)^2}$$

**Step 4:** Simplify.
$$= \frac{2x^2 - 4x - x^2 - 1}{(x-2)^2} = \frac{x^2 - 4x - 1}{(x-2)^2}$$

**Answer:** $f'(x) = \frac{x^2 - 4x - 1}{(x-2)^2}$.

### Example 4: Chain Rule

Find $f'(x)$ for $f(x) = \sin(x^2 e^x)$.

**Step 1:** Outer function $f(u) = \sin(u)$ with $u = x^2 e^x$.
**Step 2:** Derivative of outer: $f'(u) = \cos(u)$.
**Step 3:** Derivative of inner: $u' = \frac{d}{dx}[x^2 e^x] = 2x e^x + x^2 e^x = e^x (x^2 + 2x)$.
**Step 4:** Chain rule: $\frac{df}{dx} = f'(u) \cdot u'$.
$$f'(x) = \cos(x^2 e^x) \cdot e^x (x^2 + 2x)$$

**Answer:** $f'(x) = e^x (x^2 + 2x) \cos(x^2 e^x)$.

### Example 5: Derivative for Gradient Descent

Consider the loss function $L(w) = (wx - y)^2$ for a single data point $(x, y)$. Compute $\frac{dL}{dw}$.

**Step 1:** Let $u = wx - y$, then $L = u^2$.
**Step 2:** $\frac{dL}{du} = 2u = 2(wx - y)$.
**Step 3:** $\frac{du}{dw} = x$.
**Step 4:** Chain rule: $\frac{dL}{dw} = 2(wx - y) \cdot x$.

**Answer:** $\frac{dL}{dw} = 2x(wx - y)$.

**Gradient descent update:** $w_{t+1} = w_t - \alpha \cdot 2x(w_t x - y)$. This is the update for linear regression trained with MSE loss.

### Example 6: Second Derivative and Concavity

Find where $f(x) = x^3 - 6x^2 + 9x + 1$ is concave up/down.

**Step 1:** $f'(x) = 3x^2 - 12x + 9$.
**Step 2:** $f''(x) = 6x - 12$.
**Step 3:** $f''(x) = 0 \implies x = 2$.
**Step 4:** For $x < 2$: $f''(x) < 0$ → concave down. For $x > 2$: $f''(x) > 0$ → concave up.

**Answer:** Concave down on $(-\infty, 2)$, concave up on $(2, \infty)$. Inflection at $(2, 3)$.

## Visual Interpretation

**Tangent Line:**
At $x = a$, the tangent line touches the curve and has slope $f'(a)$. The equation is $y - f(a) = f'(a)(x - a)$.

**Derivative Sign:**
- $f'(x) > 0$: function increasing (sloping upward).
- $f'(x) < 0$: function decreasing (sloping downward).
- $f'(x) = 0$: horizontal tangent (potential extremum or saddle point).

**Second Derivative Sign:**
- $f''(x) > 0$: concave up (U-shaped, like a cup).
- $f''(x) < 0$: concave down (cap-shaped, like a frown).

**Gradient Descent View:**
The loss landscape is a surface. The derivative gives the slope at the current point. Moving downhill (negative gradient direction) reduces the loss.

## Common Mistakes

1. **Forgetting the chain rule.** $\frac{d}{dx} \sin(x^2) = \cos(x^2) \cdot 2x$, not just $\cos(x^2)$.

2. **Confusing $f'(g(x))$ with $(f(g(x)))'$.** $f'(g(x))$ means derivative of $f$ evaluated at $g(x)$. $(f(g(x)))' = f'(g(x)) \cdot g'(x)$.

3. **Sign errors in quotient rule.** $(\frac{u}{v})' = \frac{u'v - uv'}{v^2}$, not $\frac{uv' - u'v}{v^2}$.

4. **Treating $e^x$ as a power function.** $\frac{d}{dx}[e^x] = e^x$, not $xe^{x-1}$.

5. **Thinking $f'(c) = 0$ always means an extremum.** $f(x) = x^3$ has $f'(0) = 0$ but it is an inflection point.

6. **Forgetting the negative sign in $\frac{d}{dx}[\cos x]$.** $\frac{d}{dx}[\cos x] = -\sin x$.

7. **Using the product rule when not needed.** $f(x) = x \cdot x$ is simpler as $x^2$ with power rule.

8. **Confusing $\ln x$ and $\log_{10} x$ derivatives.** $\frac{d}{dx}[\ln x] = \frac{1}{x}$, but $\frac{d}{dx}[\log_{10} x] = \frac{1}{x \ln 10}$.

9. **Applying power rule when exponent varies.** $\frac{d}{dx}[x^n] = nx^{n-1}$ only for constant $n$. For $x^x$, use logarithmic differentiation.

10. **Ignoring domain restrictions.** $\frac{d}{dx}[\ln x] = 1/x$ exists for $x \neq 0$, but $\ln x$ itself is only defined for $x > 0$.

## Interview Questions

### Beginner

1. **What is the derivative of a function?**
   *Answer: $f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}$ is the instantaneous rate of change. Geometrically, it is the slope of the tangent line at $(x, f(x))$.*

2. **Differentiate $f(x) = 3x^4 - 2x^2 + 5$.**
   *Answer: $f'(x) = 12x^3 - 4x$.*

3. **What is the product rule?**
   *Answer: $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$.*

4. **Why is $\frac{d}{dx}[e^x] = e^x$?**
   *Answer: Using the definition: $\lim_{h\to 0} \frac{e^{x+h} - e^x}{h} = e^x \lim_{h\to 0} \frac{e^h - 1}{h} = e^x \cdot 1 = e^x$.*

5. **What does $f'(x) > 0$ tell us?**
   *Answer: $f$ is increasing on that interval.*

### Intermediate

1. **Explain how derivatives are used in gradient descent.**
   *Answer: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$. The gradient (vector of derivatives) points in the direction of steepest ascent; moving opposite to it decreases the loss. The learning rate $\alpha$ controls step size.*

2. **Differentiate $f(x) = \frac{\sin x}{x^2 + 1}$.**
   *Answer: $f'(x) = \frac{(\cos x)(x^2+1) - (\sin x)(2x)}{(x^2+1)^2}$.*

3. **How does the chain rule enable backpropagation?**
   *Answer: A network computes $L = f(f_3(f_2(f_1(x))))$. The gradient $\partial L / \partial W_1$ is a chain of derivatives: $(\partial L / \partial f_3) \cdot (\partial f_3 / \partial f_2) \cdot (\partial f_2 / \partial f_1) \cdot (\partial f_1 / \partial W_1)$. Backpropagation computes these efficiently using the chain rule.*

4. **Find the derivative of $x^x$.**
   *Answer: Use logarithmic differentiation: $\ln y = x \ln x$, so $y'/y = \ln x + 1$, giving $y' = x^x (1 + \ln x)$.*

5. **Why does the vanishing gradient problem occur with sigmoid activations?**
   *Answer: $\sigma'(z) \leq 0.25$. In deep networks, gradients are products $\prod \sigma'(z_i)$ across layers. With many layers, this product tends to 0 exponentially, making early layers learn very slowly.*

### Advanced

1. **Prove the product rule using the limit definition.**
   *Answer: $(fg)'(x) = \lim_{h\to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$. Add and subtract $f(x+h)g(x)$: $= \lim_{h\to 0} \frac{f(x+h)[g(x+h)-g(x)] + [f(x+h)-f(x)]g(x)}{h} = f(x)g'(x) + f'(x)g(x)$, provided both limits exist.*

2. **Derive the update rule for Newton's method and explain why it converges faster than gradient descent.**
   *Answer: Newton minimises a quadratic approximation: $L(\theta + \Delta\theta) \approx L(\theta) + \nabla L^T \Delta\theta + \frac{1}{2} \Delta\theta^T H \Delta\theta$. Setting derivative w.r.t. $\Delta\theta$ to 0: $H \Delta\theta = -\nabla L$, giving $\Delta\theta = -H^{-1} \nabla L$. Newton's method has quadratic convergence (near a minimum) vs. gradient descent's linear convergence, because it incorporates second-order curvature information. However, computing $H^{-1}$ is $O(n^3)$, motivating quasi-Newton methods like L-BFGS.*

3. **Explain automatic differentiation and how it differs from symbolic and numerical differentiation.**
   *Answer: (1) Numerical differentiation: $f'(x) \approx (f(x+h) - f(x))/h$ is approximate and suffers from round-off error. (2) Symbolic differentiation: manipulates algebraic expressions (like $\frac{d}{dx}[x^2] \to 2x$), but can produce exponentially large expressions. (3) Automatic differentiation (AD): decomposes functions into elementary operations and applies the chain rule numerically at each step. AD computes exact derivatives (to machine precision) in $O(1)$ times the cost of the original function. Forward-mode AD propagates derivatives alongside the computation (good for few inputs, many outputs). Reverse-mode AD (backpropagation) propagates derivatives backward (good for many inputs, few outputs — exactly what neural networks need).*

## Practice Problems

### Easy

1. Find $f'(x)$ for $f(x) = 5x^3 - 2x + 7$.
2. Differentiate $f(x) = e^x \cos x$.
3. Find $\frac{d}{dx} \ln(3x^2 + 1)$.
4. Compute $f'(2)$ for $f(x) = x^3 - 4x$.
5. Find $\frac{d}{dx} \sin(2x)$.

### Medium

1. Differentiate $f(x) = \frac{x^3}{e^x}$.
2. Find $f'(x)$ for $f(x) = \sin^2(x) \cos(x)$.
3. Compute $\frac{d}{dx} \arctan(x^2)$.
4. Find the equation of the tangent line to $y = \ln x$ at $x = e$.
5. If $L(w) = (wx - y)^2 + \lambda w^2$, compute $\frac{dL}{dw}$.

### Hard

1. Derive Leibniz's formula for the $n$-th derivative of a product: $(f \cdot g)^{(n)} = \sum_{k=0}^n \binom{n}{k} f^{(k)} g^{(n-k)}$.
2. Prove the chain rule using Carath\'eodory's lemma: $f$ is differentiable at $g(x)$ with derivative $f'(g(x))$ iff there exists $\phi$ continuous at $g(x)$ such that $f(y) - f(g(x)) = \phi(y)(y - g(x))$, with $\phi(g(x)) = f'(g(x))$.
3. Design a custom activation function whose derivative is always positive (to avoid dead ReLU) and compute its derivative explicitly.

## Solutions

### Easy Solutions

**1.** $f'(x) = 15x^2 - 2$.

**2.** Product rule: $f'(x) = e^x \cos x + e^x (-\sin x) = e^x(\cos x - \sin x)$.

**3.** Chain rule: $\frac{1}{3x^2+1} \cdot 6x = \frac{6x}{3x^2+1}$.

**4.** $f'(x) = 3x^2 - 4$, so $f'(2) = 3(4) - 4 = 8$.

**5.** Chain rule: $\cos(2x) \cdot 2 = 2\cos(2x)$.

### Medium Solutions

**1.** Quotient rule: $f'(x) = \frac{3x^2 \cdot e^x - x^3 \cdot e^x}{(e^x)^2} = \frac{3x^2 - x^3}{e^x} = \frac{x^2(3 - x)}{e^x}$.

**2.** Product rule with chain rule: $f'(x) = 2\sin x \cos x \cdot \cos x + \sin^2 x \cdot (-\sin x) = 2\sin x \cos^2 x - \sin^3 x$.

**3.** Chain rule: $\frac{1}{1 + (x^2)^2} \cdot 2x = \frac{2x}{1 + x^4}$.

**4.** $y' = 1/x$, so at $x = e$, slope $= 1/e$. Point: $(e, 1)$. Equation: $y - 1 = (1/e)(x - e)$, i.e., $y = x/e$.

**5.** $\frac{dL}{dw} = 2(wx - y) \cdot x + 2\lambda w = 2x(wx - y) + 2\lambda w = 2w(x^2 + \lambda) - 2xy$.

### Hard Solutions

**1.** Proof by induction. Base $n = 1$: $(fg)' = f'g + fg'$ — the product rule. Inductive step: assume true for $n$. Then $(fg)^{(n+1)} = \frac{d}{dx} \sum_{k=0}^n \binom{n}{k} f^{(k)} g^{(n-k)} = \sum_{k=0}^n \binom{n}{k} (f^{(k+1)} g^{(n-k)} + f^{(k)} g^{(n-k+1)})$. Reindexing and using Pascal's identity $\binom{n}{k} + \binom{n}{k-1} = \binom{n+1}{k}$ gives $\sum_{k=0}^{n+1} \binom{n+1}{k} f^{(k)} g^{(n+1-k)}$.

**2.** Carath\'eodory lemma: $f$ is differentiable at $x_0$ with $f'(x_0)$ iff there exists a function $\phi$ continuous at $x_0$ such that $f(x) - f(x_0) = \phi(x)(x - x_0)$ with $\phi(x_0) = f'(x_0)$. For the chain rule: let $y_0 = g(x_0)$. Since $g$ is differentiable at $x_0$, $g(x) - g(x_0) = \psi(x)(x - x_0)$ with $\psi(x_0) = g'(x_0)$. Since $f$ is differentiable at $y_0$, $f(y) - f(y_0) = \phi(y)(y - y_0)$ with $\phi(y_0) = f'(y_0)$. Then $f(g(x)) - f(g(x_0)) = \phi(g(x))(g(x) - g(x_0)) = \phi(g(x))\psi(x)(x - x_0)$. Since $\phi(g(\cdot)) \cdot \psi(\cdot)$ is continuous at $x_0$ with value $f'(g(x_0))g'(x_0)$, the chain rule follows.

**3.** The Swish activation $f(x) = x \cdot \sigma(\beta x)$ where $\sigma$ is the sigmoid has derivative $f'(x) = \sigma(\beta x) + \beta x \sigma(\beta x)(1 - \sigma(\beta x))$, which is always positive for $\beta > 0$. Alternatively, ELU: $f(x) = \begin{cases} x & x \geq 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}$ has derivative $f'(x) = \begin{cases} 1 & x > 0 \\ \alpha e^x & x < 0 \end{cases}$, always positive. A novel activation: $f(x) = x + \frac{1}{2}\sin(x)$ has derivative $f'(x) = 1 + \frac{1}{2}\cos(x) \geq \frac{1}{2} > 0$, avoiding dead neurons entirely.

## Related Concepts

- **Limits** (MATH-053) — The derivative is defined as a limit.
- **Continuity** (MATH-054) — Differentiability implies continuity.
- **Partial Derivative** (MATH-056) — Generalisation to multivariable functions.
- **Chain Rule** (MATH-057) — Essential for backpropagation and gradient computation.
- **Gradient** (MATH-058) — Vector of partial derivatives; direction of steepest ascent.
- **Composite Function** (MATH-047) — The chain rule differentiates composites.
- **Exponential Function** (MATH-050) — Derivative of $e^x$ is itself.

## Next Concepts

- **Taylor Series** — Representing functions as infinite polynomials using higher-order derivatives.
- **Differential Equations** — Equations involving derivatives; modelling dynamic systems.
- **Automatic Differentiation** — Computational technique for evaluating derivatives efficiently.
- **Calculus of Variations** — Optimising functionals (functions of functions).

## Summary

The derivative $f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}$ measures the instantaneous rate of change of a function, geometrically interpreted as the slope of the tangent line. Differentiation rules (power, product, quotient, chain rules) enable systematic computation of derivatives for any combination of elementary functions. Higher-order derivatives capture acceleration and concavity. In AI/ML, derivatives are the foundation of gradient descent, backpropagation, and virtually all optimisation algorithms. The vanishing gradient problem — where products of small derivatives cause gradients to vanish in deep networks — is a key challenge that derivative analysis helps explain and mitigate.

## Key Takeaways

- The derivative is the slope of the tangent line: $f'(x) = \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}$.
- Differentiability implies continuity; the converse is false.
- Product rule: $(fg)' = f'g + fg'$. Quotient rule: $(\frac{u}{v})' = \frac{u'v - uv'}{v^2}$.
- Chain rule: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ — the backbone of backpropagation.
- $\frac{d}{dx}[e^x] = e^x$, $\frac{d}{dx}[\ln x] = 1/x$, $\frac{d}{dx}[\sin x] = \cos x$.
- $f'(c) = 0$ is necessary but not sufficient for an extremum.
- Gradient descent: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$.
- Vanishing gradients occur when products of small derivatives (e.g., sigmoid) dampen gradient flow.
- Higher-order derivatives: $f''$ measures concavity; used in Newton's method.
- Automatic differentiation computes exact derivatives efficiently via the chain rule.
