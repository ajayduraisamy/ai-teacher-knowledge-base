# Concept: Inverse Function

## Concept ID

MATH-048

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define the inverse function $f^{-1}$ and understand the condition $f^{-1}(f(x)) = x$ and $f(f^{-1}(y)) = y$.
- Determine whether a function has an inverse using the horizontal line test (one-to-one requirement).
- Find the inverse of a function algebraically by swapping $x$ and $y$ and solving for $y$.
- Verify inverse relationships through function composition.
- Restrict the domain of a function to make it invertible.
- Connect inverse functions to AI/ML concepts including invertible neural networks, normalizing flows, and decoding in autoencoders.

## Prerequisites

- Solid understanding of functions (MATH-044) — domain, codomain, range, and function notation.
- Understanding of domain (MATH-045) — the domain of $f^{-1}$ is the range of $f$.
- Understanding of range (MATH-046) — the range of $f^{-1}$ is the domain of $f$.
- Understanding of composite functions (MATH-047) — inverses are defined by $f(f^{-1}(x)) = f^{-1}(f(x)) = x$.
- Basic algebra: solving equations for a variable, working with fractions, exponents, and radicals.

## Definition

Let $f: A \to B$ be a function. If there exists a function $f^{-1}: B \to A$ such that:
$$f^{-1}(f(x)) = x \quad \text{for all } x \in A$$
$$f(f^{-1}(y)) = y \quad \text{for all } y \in B$$
then $f^{-1}$ is called the **inverse** of $f$. The function $f$ is then said to be **invertible**.

For $f^{-1}$ to exist, $f$ must be **bijective** — both injective (one-to-one) and surjective (onto). In practice, this means:
1. **Injectivity (One-to-one):** Different inputs map to different outputs: $x_1 \neq x_2 \implies f(x_1) \neq f(x_2)$. Equivalently, $f(x_1) = f(x_2) \implies x_1 = x_2$.
2. **Surjectivity (Onto):** Every element of the codomain is actually attained by some input.

If $f$ is not one-to-one, we can sometimes restrict its domain to make it one-to-one, and then define an inverse on that restricted domain.

The notation $f^{-1}$ should not be confused with $\frac{1}{f(x)}$ — they are completely different. $f^{-1}$ is the inverse function, while $(f(x))^{-1} = \frac{1}{f(x)}$ is the reciprocal.

## Intuition

Think of an inverse function as an **undo button**. If $f$ transforms an input into an output, then $f^{-1}$ transforms it back:

$$x \xrightarrow{f} f(x) \xrightarrow{f^{-1}} x$$

The function $f$ does something, and $f^{-1}$ undoes it. Together, they compose to the identity.

Analogies:
- **Encryption/Decryption:** $f$ encrypts a message, $f^{-1}$ decrypts it back to the original.
- **Encoding/Decoding:** $f$ encodes data into a compressed representation, $f^{-1}$ decodes it back.
- **Lock and Key:** $f$ locks a door (input $\to$ locked state), $f^{-1}$ unlocks it.

A key intuition: **not all functions have inverses**. A function that is not one-to-one is like a blender — multiple different inputs produce the same output (smoothie), and you cannot reverse the process uniquely.

## Why This Concept Matters

Inverse functions are fundamental across mathematics and AI:

**1. Solving Equations.** To solve $f(x) = y$ for $x$, we apply $f^{-1}$ to both sides: $x = f^{-1}(y)$. Every equation-solving technique is essentially about finding the inverse of a function.

**2. Cryptography.** Encryption functions must have inverses (decryption keys) that are hard to discover without the key.

**3. Machine Learning — Decoding.** Autoencoders consist of an encoder $E$ that compresses data and a decoder $D$ that reconstructs it. $D$ approximates the inverse of $E$.

**4. Normalizing Flows.** A normalizing flow is a composition of invertible functions $f = f_K \circ \cdots \circ f_1$. Because each $f_i$ is invertible, the entire flow is invertible, enabling exact likelihood computation.

**5. Change of Variables.** In probability, if $Y = f(X)$ and $f$ is invertible and differentiable, the density of $Y$ is given by:
$$p_Y(y) = p_X(f^{-1}(y)) \cdot \left|\frac{d}{dy}f^{-1}(y)\right|$$

**6. Inverse Problems.** Many scientific problems involve inferring causes from effects — the inverse of a forward model. For example, medical imaging reconstructs tissue properties from sensor measurements.

**7. Optimization.** In gradient-based optimization, we sometimes need to compute the inverse of a matrix (e.g., Newton method uses the inverse Hessian).

## Historical Background

The concept of inverse functions dates back to the earliest days of calculus. Isaac Newton and Gottfried Wilhelm Leibniz used inverse operations (differentiation and integration are inverse operations — the Fundamental Theorem of Calculus). In the 18th century, Leonhard Euler studied inverse trigonometric functions and developed notations like $\arcsin$ and $\arccos$. The modern definition of an inverse function was formalized in the 19th century by Cauchy and Weierstrass. The notation $f^{-1}$ was popularized by John Herschel in the early 19th century.

## Real World Examples

**Example 1: Celsius and Fahrenheit.** $C(F) = \frac{5}{9}(F - 32)$ converts Fahrenheit to Celsius. Its inverse $F(C) = \frac{9}{5}C + 32$ converts Celsius back to Fahrenheit.

**Example 2: Distance and Time.** At constant speed $v = 60$ km/h, distance is $d(t) = 60t$. The inverse $t(d) = \frac{d}{60}$ gives the time needed.

**Example 3: Loan Payments.** For a fixed-rate loan, the monthly payment $M(P)$ is a function of the principal $P$. The inverse $P(M)$ tells the maximum loan for a given monthly payment.

**Example 4: Log and Exponential.** $f(x) = e^x$ and $f^{-1}(x) = \ln x$ are inverses.

**Example 5: Shipping and Returns.** A shipping company delivers a package (function $f$). The return process is the inverse $f^{-1}$. Together they compose to the identity.

## AI/ML Relevance

Inverse functions have profound applications in AI and machine learning:

**1. Normalizing Flows.** A normalizing flow is a generative model that learns a transformation $f$ from a simple base distribution (e.g., Gaussian) to a complex data distribution. The key insight is that $f$ must be **invertible** and **differentiable**. The density of a data point $x$ is:
$$p_X(x) = p_Z(f^{-1}(x)) \cdot \left|\det J_{f^{-1}}(x)\right|$$
where $Z \sim \mathcal{N}(0, I)$. Flow-based models like Glow and RealNVP are built on this principle.

**2. Autoencoders and VAEs.** An autoencoder learns $E: X \to Z$ (encoder) and $D: Z \to X$ (decoder) such that $D(E(x)) \approx x$. Ideally, $D \approx E^{-1}$. In practice, $E$ is not exactly invertible (information is lost), but the decoder learns a pseudo-inverse.

**3. Invertible Neural Networks.** Some architectures (e.g., i-RevNet, ResNet with invertible blocks) are designed to be fully invertible. This allows memory-efficient training (activations reconstructed from outputs during backpropagation) and exact likelihood computation.

**4. Change of Variables in Probabilistic Models.** When transforming random variables, the inverse function appears in the density transformation formula:
$$p_Y(y) = p_X(f^{-1}(y)) \cdot \left|\det \frac{\partial f^{-1}(y)}{\partial y}\right|$$

**5. Inverse Reinforcement Learning (IRL).** IRL infers the reward function that an agent is optimizing from observed behavior. If the forward problem is "given rewards, find optimal policy," then IRL solves the inverse problem.

**6. Image Generation and Manipulation.** GANs and diffusion models learn generative processes. The inverse of the generation process can be used for image editing, style transfer, and compression.

**7. Gradient-based Optimization.** Newton method uses the inverse Hessian matrix to accelerate convergence. Quasi-Newton methods (BFGS, L-BFGS) approximate the inverse Hessian.

## Mathematical Explanation

**Finding the Inverse Algebraically:**

To find the inverse of a function $f(x)$:
1. Write $y = f(x)$.
2. Swap $x$ and $y$: $x = f(y)$.
3. Solve for $y$ in terms of $x$.
4. The resulting expression is $y = f^{-1}(x)$.
5. Verify: check $f(f^{-1}(x)) = x$ and $f^{-1}(f(x)) = x$.

**Example:** Find $f^{-1}(x)$ for $f(x) = 2x + 3$.
1. $y = 2x + 3$.
2. Swap: $x = 2y + 3$.
3. Solve: $2y = x - 3 \implies y = \frac{x - 3}{2}$.
4. $f^{-1}(x) = \frac{x - 3}{2}$.
5. Verify: $f(f^{-1}(x)) = 2(\frac{x-3}{2}) + 3 = x$ and $f^{-1}(f(x)) = \frac{(2x+3)-3}{2} = x$.

**Horizontal Line Test:**

A function $f$ has an inverse if and only if every horizontal line intersects the graph of $f$ at most once. This is equivalent to $f$ being **one-to-one** (injective).

**One-to-One Functions:**

$f$ is one-to-one if $f(x_1) = f(x_2) \implies x_1 = x_2$. Strictly monotonic functions (always increasing or always decreasing) are one-to-one.

**Restricting Domain to Make a Function Invertible:**

Many functions are not one-to-one on their natural domain but become one-to-one on a restricted domain:
- $f(x) = x^2$ on $[0, \infty)$ with inverse $f^{-1}(x) = \sqrt{x}$.
- $f(x) = \sin x$ on $[-\frac{\pi}{2}, \frac{\pi}{2}]$ with inverse $\sin^{-1}(x) = \arcsin(x)$.
- $f(x) = \cos x$ on $[0, \pi]$ with inverse $\cos^{-1}(x) = \arccos(x)$.

**Graph of the Inverse:**

The graph of $f^{-1}$ is the reflection of the graph of $f$ across the line $y = x$.

**Inverse of a Composition:**

$(f \circ g)^{-1} = g^{-1} \circ f^{-1}$.

**Derivative of the Inverse:**

If $f$ is differentiable and $f'(x) \neq 0$, then:
$$(f^{-1})'(y) = \frac{1}{f'(f^{-1}(y))} = \frac{1}{f'(x)} \quad \text{where} \quad y = f(x)$$

## Properties

1. **Uniqueness:** If an inverse exists, it is unique.
2. **Self-inverse (involution):** If $f = f^{-1}$, then $f$ is an involution. Examples: $f(x) = -x$, $f(x) = 1/x$, $f(x) = a - x$.
3. **Symmetry:** $f$ and $f^{-1}$ are symmetric about $y = x$ on the graph.
4. **Reversal of composition:** $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$.
5. **Domain and range swap:** $\text{dom}(f^{-1}) = \text{range}(f)$ and $\text{range}(f^{-1}) = \text{dom}(f)$.
6. **Inverse of inverse:** $(f^{-1})^{-1} = f$.
7. **Monotonicity:** If $f$ is strictly increasing, then $f^{-1}$ is strictly increasing. If $f$ is strictly decreasing, then $f^{-1}$ is strictly decreasing.
8. **Derivative relationship:** $(f^{-1})'(f(x)) = \frac{1}{f'(x)}$ (provided $f'(x) \neq 0$).
9. **Only bijections have inverses:** A function must be both injective and surjective to have an inverse defined on the entire codomain.

## Formula(s)

**Definition (cancellation equations):**
$$f^{-1}(f(x)) = x, \quad f(f^{-1}(y)) = y$$

**Domain and range of inverse:**
$$\text{dom}(f^{-1}) = \text{range}(f), \quad \text{range}(f^{-1}) = \text{dom}(f)$$

**Inverse of a linear function:**
$$f(x) = mx + b \;(m \neq 0) \quad \implies \quad f^{-1}(x) = \frac{x - b}{m}$$

**Inverse of exponential and logarithm:**
$$f(x) = b^x \;\iff\; f^{-1}(x) = \log_b(x)$$

**Inverse of power functions:**
$$f(x) = x^n \;(n \text{ odd}) \;\implies\; f^{-1}(x) = \sqrt[n]{x}$$
$$f(x) = x^n \;(n \text{ even, on }[0,\infty)) \;\implies\; f^{-1}(x) = \sqrt[n]{x}$$

**Inverse of trigonometric functions (restricted domains):**
$$\sin^{-1}(x) = \arcsin(x) \text{ on } [-\tfrac{\pi}{2}, \tfrac{\pi}{2}], \quad \cos^{-1}(x) = \arccos(x) \text{ on } [0, \pi], \quad \tan^{-1}(x) = \arctan(x) \text{ on } (-\tfrac{\pi}{2}, \tfrac{\pi}{2})$$

**Inverse of a composition:**
$$(f \circ g)^{-1} = g^{-1} \circ f^{-1}$$

**Derivative of the inverse:**
$$(f^{-1})'(y) = \frac{1}{f'(f^{-1}(y))} = \frac{1}{f'(x)} \quad \text{where} \quad y = f(x)$$

## Step-by-Step Worked Examples

### Example 1: Inverse of a Linear Function

Find the inverse of $f(x) = 4x - 7$.

**Step 1:** Write $y = f(x)$: $y = 4x - 7$.

**Step 2:** Swap $x$ and $y$: $x = 4y - 7$.

**Step 3:** Solve for $y$: $4y = x + 7$, $y = \frac{x + 7}{4}$.

**Step 4:** Verify: $f(f^{-1}(x)) = 4(\frac{x+7}{4}) - 7 = x$, $f^{-1}(f(x)) = \frac{(4x-7)+7}{4} = \frac{4x}{4} = x$.

**Answer:** $f^{-1}(x) = \frac{x + 7}{4}$

### Example 2: Inverse of a Rational Function

Find the inverse of $f(x) = \frac{2x + 1}{x - 3}$.

**Step 1:** Write $y = \frac{2x + 1}{x - 3}$.

**Step 2:** Swap $x$ and $y$: $x = \frac{2y + 1}{y - 3}$.

**Step 3:** Solve for $y$:
$x(y - 3) = 2y + 1 \implies xy - 3x = 2y + 1 \implies xy - 2y = 3x + 1 \implies y(x - 2) = 3x + 1 \implies y = \frac{3x + 1}{x - 2}$.

**Step 4:** Domain/range check: original $f$ has domain $x \neq 3$, range $y \neq 2$. Inverse has domain $x \neq 2$, range $y \neq 3$.

**Answer:** $f^{-1}(x) = \frac{3x + 1}{x - 2}$

### Example 3: Inverse with Domain Restriction

Find the inverse of $f(x) = x^2 - 4x + 5$, $x \geq 2$.

**Step 1:** Write $y = x^2 - 4x + 5 = (x - 2)^2 + 1$.

**Step 2:** Swap $x$ and $y$: $x = (y - 2)^2 + 1$.

**Step 3:** Solve: $(y - 2)^2 = x - 1$, $y - 2 = \sqrt{x - 1}$ (since $y \geq 2$), $y = 2 + \sqrt{x - 1}$.

**Step 4:** Domain of inverse: $x \geq 1$ (from square root).

**Answer:** $f^{-1}(x) = 2 + \sqrt{x - 1}$, $x \geq 1$

### Example 4: Inverse of an Exponential Function

Find the inverse of $f(x) = 2e^{3x} + 1$.

**Step 1:** Write $y = 2e^{3x} + 1$.

**Step 2:** Swap: $x = 2e^{3y} + 1$.

**Step 3:** Solve: $x - 1 = 2e^{3y}$, $\frac{x-1}{2} = e^{3y}$, $3y = \ln(\frac{x-1}{2})$, $y = \frac{1}{3}\ln(\frac{x-1}{2})$.

**Step 4:** Domain of inverse: $x > 1$.

**Answer:** $f^{-1}(x) = \frac{1}{3}\ln(\frac{x-1}{2})$, $x > 1$

### Example 5: Verifying Inverses via Composition

Verify $f(x) = \sqrt[3]{x - 1}$ and $g(x) = x^3 + 1$ are inverses.

**Step 1:** $f(g(x)) = \sqrt[3]{(x^3 + 1) - 1} = \sqrt[3]{x^3} = x$.

**Step 2:** $g(f(x)) = (\sqrt[3]{x - 1})^3 + 1 = x - 1 + 1 = x$.

**Answer:** Verified.

### Example 6: Derivative of an Inverse

Given $f(x) = x^3 + 2x$, find $(f^{-1})'(3)$.

**Step 1:** $f(1) = 1 + 2 = 3$, so $f^{-1}(3) = 1$.

**Step 2:** $f'(x) = 3x^2 + 2$, so $f'(1) = 5$.

**Step 3:** $(f^{-1})'(3) = \frac{1}{f'(1)} = \frac{1}{5}$.

**Answer:** $(f^{-1})'(3) = \frac{1}{5}$

### Example 7: Inverse in Normalizing Flows

A simple affine coupling layer: $f(x_1, x_2) = (x_1, x_2 e^{x_1})$. Find $f^{-1}$.

**Step 1:** Forward: $y_1 = x_1$, $y_2 = x_2 e^{x_1}$.

**Step 2:** Invert: $x_1 = y_1$, $x_2 = y_2 e^{-x_1} = y_2 e^{-y_1}$.

**Step 3:** $f^{-1}(y_1, y_2) = (y_1, y_2 e^{-y_1})$.

**Step 4:** Verify: $f(f^{-1}(y)) = (y_1, y_2 e^{-y_1} \cdot e^{y_1}) = (y_1, y_2)$. ✓

## Visual Interpretation

**Reflection Across $y = x$:**

The graph of $f^{-1}$ is obtained by reflecting the graph of $f$ across the line $y = x$. This is the most important visual representation of inverse functions. If you fold the graph along $y = x$, the function and its inverse overlap perfectly.

**Horizontal Line Test:**

To determine whether $f$ has an inverse, draw horizontal lines. If any horizontal line hits the graph more than once, $f$ does not have an inverse. The parabola $y = x^2$ fails the horizontal line test because $y = 4$ hits at $x = 2$ and $x = -2$.

**Mapping Diagram:**

A one-to-one function has bi-directional arrows in its mapping diagram:
```
Domain    Range
  1  --►    a
  2  --►    b
  3  --►    c
```
The inverse reverses the arrows:
```
Domain    Range
  a  --►    1
  b  --►    2
  c  --►    3
```

## Common Mistakes

1. **Confusing $f^{-1}$ with $\frac{1}{f}$.** $f^{-1}$ is the inverse function (undoes $f$), while $\frac{1}{f(x)} = (f(x))^{-1}$ is the reciprocal. For example, $\sin^{-1}(x)$ is arcsine, not $\csc(x) = 1/\sin(x)$.

2. **Assuming every function has an inverse.** Only one-to-one (injective) functions have inverses. $f(x) = x^2$ on $\mathbb{R}$ has no inverse.

3. **Forgetting to restrict the domain when necessary.** $f(x) = x^2$ has an inverse if restricted to $[0, \infty)$. Without specifying the restriction, $f^{-1}(4) = \pm 2$ is ambiguous.

4. **Incorrectly solving for the inverse.** A common algebraic error is swapping $x$ and $y$ but then solving for $x$ instead of $y$, or failing to treat $y$ as the variable to isolate.

5. **Getting the composition order wrong for inverse of composition.** $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$, not $f^{-1} \circ g^{-1}$.

6. **Assuming $f(f^{-1}(x)) = x$ for all $x$.** This holds only for $x$ in the domain of $f^{-1}$ (range of $f$).

7. **Solving for $x$ without swapping variables first.** The correct procedure: swap $x$ and $y$ first, then solve. If you solve $y = f(x)$ for $x$ without swapping, you get $x = f^{-1}(y)$ (inverse with $y$ as input).

8. **Assuming one-to-one implies monotonic.** A function can be one-to-one without being monotonic if it is not continuous. Every continuous one-to-one function on an interval is monotonic.

## Interview Questions

### Beginner

1. **What is an inverse function?**
   *Answer: An inverse function $f^{-1}$ undoes the action of $f$. It satisfies $f^{-1}(f(x)) = x$ for all $x$ in the domain of $f$, and $f(f^{-1}(y)) = y$ for all $y$ in the domain of $f^{-1}$.*

2. **What is the horizontal line test?**
   *Answer: It checks whether a function is one-to-one. If any horizontal line intersects the graph at more than one point, the function is not one-to-one and has no inverse.*

3. **Find the inverse of $f(x) = 3x - 5$.**
   *Answer: $y = 3x - 5$. Swap: $x = 3y - 5$. Solve: $y = \frac{x+5}{3}$. So $f^{-1}(x) = \frac{x+5}{3}$.*

4. **Why does $f(x) = x^2$ not have an inverse on $\mathbb{R}$?**
   *Answer: It is not one-to-one on $\mathbb{R}$ ($f(2) = f(-2) = 4$). Restrict to $[0, \infty)$ to get inverse $f^{-1}(x) = \sqrt{x}$.*

5. **What is the relationship between the graphs of $f$ and $f^{-1}$?**
   *Answer: The graph of $f^{-1}$ is the reflection of the graph of $f$ across the line $y = x$.*

6. **How can you verify two functions are inverses?**
   *Answer: Compute both compositions $f(g(x))$ and $g(f(x))$. If both equal $x$, they are inverses.*

### Intermediate

1. **Find the inverse of $f(x) = \frac{2x}{x - 1}$. State its domain and range.**
   *Answer: $f^{-1}(x) = \frac{x}{x-2}$, domain $x \neq 2$, range $y \neq 1$.*

2. **Why must a neural network in a normalizing flow be invertible?**
   *Answer: Invertibility enables exact likelihood computation via the change-of-variable formula. Sampling goes forward ($z \to x$), density evaluation goes backward ($x \to z$). Without invertibility, the mapping is not one-to-one and the density transformation is undefined.*

3. **Find the inverse of $f(x) = \sqrt{x - 2} + 3$ and its domain.**
   *Answer: $f^{-1}(x) = (x - 3)^2 + 2$, $x \geq 3$.*

4. **Prove that $(f^{-1})'(f(x)) = \frac{1}{f'(x)}$.**
   *Proof: Differentiate $f^{-1}(f(x)) = x$ using chain rule: $(f^{-1})'(f(x)) \cdot f'(x) = 1$. Therefore $(f^{-1})'(f(x)) = 1/f'(x)$.*

5. **How does the decoder in an autoencoder relate to the concept of an inverse function?**
   *Answer: The decoder $D$ approximates the left-inverse of the encoder $E$: $D(E(x)) \approx x$. However, due to the bottleneck, $E$ is not injective and $D$ is not surjective, so they are pseudo-inverses rather than true inverses.*

6. **Given $f(x) = x^3 - 1$, find $f^{-1}(x)$ and compute $(f^{-1})'(7)$.**
   *Answer: $f^{-1}(x) = \sqrt[3]{x + 1}$. $f^{-1}(7) = 2$, $f'(2) = 12$, so $(f^{-1})'(7) = 1/12$.*

### Advanced

1. **Prove that if $f$ is continuous and strictly monotonic on $I$, then $f^{-1}$ is also continuous and strictly monotonic on $f(I)$.**
   *Proof: Strict monotonic $\implies$ injective $\implies$ $f^{-1}$ exists. If $y_1 < y_2$ in $f(I)$, then $f^{-1}(y_1) < f^{-1}(y_2)$, else applying $f$ would give a contradiction, so $f^{-1}$ is strictly monotonic. Continuity follows from the intermediate value property of continuous monotonic functions on intervals.*

2. **Explain the role of invertible functions in the change-of-variable formula for probability densities and how this relates to normalizing flows.**
   *Answer: If $X = f(Z)$ with $f$ invertible, then $p_X(x) = p_Z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)| = p_Z(z) \cdot |\det J_f(z)|^{-1}$. In normalizing flows, $f$ is composed of simple invertible layers with triangular Jacobians, making the log-determinant tractable. This is why invertibility is essential: without it, we cannot compute the density of $X$ from $Z$.*

3. **Prove that $f(x) = x + \sin x$ is invertible on $\mathbb{R}$ and find an expression for $(f^{-1})'(y)$.**
   *Proof: $f'(x) = 1 + \cos x \geq 0$, zero only at isolated points $x = \pi + 2\pi k$. Since $f'(x) > 0$ on intervals between these points and $f$ is continuous, $f$ is strictly increasing, hence injective. As $x \to \pm\infty$, $f(x) \to \pm\infty$, so $f$ is surjective. Therefore $f$ is bijective and invertible. $(f^{-1})'(y) = \frac{1}{1 + \cos(f^{-1}(y))}$.*

## Practice Problems

### Easy

1. Find the inverse of $f(x) = 5x + 2$.
2. Find the inverse of $f(x) = \frac{x}{2} - 3$.
3. Use the horizontal line test to determine whether $f(x) = |x|$ has an inverse.
4. Find the inverse of $f(x) = \sqrt{x + 4}$.
5. Verify that $f(x) = 2x - 5$ and $g(x) = \frac{x + 5}{2}$ are inverses.

### Medium

1. Find the inverse of $f(x) = \frac{3x - 1}{x + 2}$.
2. Find the inverse of $f(x) = x^2 - 2x$ with domain $x \geq 1$.
3. Find $f^{-1}(x)$ for $f(x) = e^{2x - 1}$.
4. If $f(x) = 2x + 1$ and $g(x) = x^2$, find $(f \circ g)^{-1}(x)$.
5. Given $f(x) = \ln(x + 2)$, find $f^{-1}(x)$ and its domain.

### Hard

1. Let $f(x) = \frac{ax + b}{cx + d}$ with $ad - bc \neq 0$. Prove $f^{-1}(x) = \frac{dx - b}{-cx + a}$.
2. Show $f(x) = x^3 + 3x$ is invertible on $\mathbb{R}$ and find $(f^{-1})'(4)$.
3. A bijective function $f$ satisfies $(f \circ f)(x) = 2x - 1$ for all $x$. Find $f(x)$ and $f^{-1}(x)$.

## Solutions

### Easy Solutions

**1.** $y = 5x + 2$, swap: $x = 5y + 2$, $5y = x - 2$, $y = \frac{x - 2}{5}$. $f^{-1}(x) = \frac{x - 2}{5}$.

**2.** $y = \frac{x}{2} - 3$, swap: $x = \frac{y}{2} - 3$, $\frac{y}{2} = x + 3$, $y = 2x + 6$. $f^{-1}(x) = 2x + 6$.

**3.** $f(x) = |x|$ fails the horizontal line test ($y = 2$ hits at $x = 2$ and $x = -2$). No inverse.

**4.** $y = \sqrt{x + 4}$, domain $x \geq -4$. Swap: $x = \sqrt{y + 4}$. $x^2 = y + 4$, $y = x^2 - 4$. Domain of inverse: $x \geq 0$. $f^{-1}(x) = x^2 - 4$, $x \geq 0$.

**5.** $f(g(x)) = 2(\frac{x+5}{2}) - 5 = x$. $g(f(x)) = \frac{(2x-5)+5}{2} = x$. Verified.

### Medium Solutions

**1.** $y = \frac{3x - 1}{x + 2}$, swap: $x = \frac{3y - 1}{y + 2}$. $x(y+2) = 3y - 1 \implies xy + 2x = 3y - 1 \implies xy - 3y = -2x - 1 \implies y(x-3) = -(2x+1) \implies y = \frac{2x+1}{3-x}$. $f^{-1}(x) = \frac{2x+1}{3-x}$.

**2.** $f(x) = (x-1)^2 - 1$, $x \geq 1$. Swap: $x = (y-1)^2 - 1$. $(y-1)^2 = x+1$, $y-1 = \sqrt{x+1}$, $y = 1 + \sqrt{x+1}$. Range of $f$: $[-1, \infty)$. $f^{-1}(x) = 1 + \sqrt{x+1}$, $x \geq -1$.

**3.** $y = e^{2x-1}$, swap: $x = e^{2y-1}$. $\ln x = 2y-1$, $2y = \ln x + 1$, $y = \frac{\ln x + 1}{2}$. $f^{-1}(x) = \frac{\ln x + 1}{2}$, $x > 0$.

**4.** $(f \circ g)(x) = 2x^2 + 1$. $y = 2x^2 + 1$, swap: $x = 2y^2 + 1$. $y^2 = \frac{x-1}{2}$, $y = \sqrt{\frac{x-1}{2}}$. $(f \circ g)^{-1}(x) = \sqrt{\frac{x-1}{2}}$, $x \geq 1$.

**5.** $y = \ln(x+2)$, swap: $x = \ln(y+2)$. $e^x = y+2$, $y = e^x - 2$. Range of $f$: $\mathbb{R}$, so $f^{-1}(x) = e^x - 2$, domain $\mathbb{R}$.

### Hard Solutions

**1.** $y = \frac{ax+b}{cx+d}$. Swap: $x = \frac{ay+b}{cy+d}$. $x(cy+d) = ay+b \implies cxy + dx = ay+b \implies cxy - ay = b - dx \implies y(cx - a) = b - dx \implies y = \frac{b - dx}{cx - a} = \frac{dx - b}{-cx + a}$. The condition $ad - bc \neq 0$ ensures the denominator is not identically zero.

**2.** $f'(x) = 3x^2 + 3 > 0$ for all $x$, so $f$ is strictly increasing $\implies$ injective. As $x \to \pm\infty$, $f(x) \to \pm\infty$, so $f$ is surjective. Hence $f$ is bijective and invertible. $f(1) = 4$, so $f^{-1}(4) = 1$. $f'(1) = 6$. Therefore $(f^{-1})'(4) = 1/6$.

**3.** Let $f(x) = ax + b$ (linear). Then $f(f(x)) = a(ax+b)+b = a^2 x + ab + b = 2x - 1$. So $a^2 = 2 \implies a = \sqrt{2}$ (positive for bijection on $\mathbb{R}$) and $b(a+1) = -1 \implies b = \frac{-1}{\sqrt{2}+1} = 1 - \sqrt{2}$. So $f(x) = \sqrt{2}x + 1 - \sqrt{2}$. Then $f^{-1}(x) = \frac{x + \sqrt{2} - 1}{\sqrt{2}}$.

## Related Concepts

- **Function** (MATH-044) — The inverse is a special type of function that undoes another function.
- **Domain** (MATH-045) — The domain of $f^{-1}$ is the range of $f$.
- **Range** (MATH-046) — The range of $f^{-1}$ is the domain of $f$.
- **Composite Function** (MATH-047) — $f^{-1} \circ f = id$, $f \circ f^{-1} = id$, $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$.
- **One-to-One Functions (Injectivity)** — A function must be injective to have an inverse.
- **Horizontal Line Test** — Graphical test for the one-to-one property.
- **Reflection Across $y = x$** — The graph of an inverse is the reflection of the original.
- **Monotonic Functions** — Strictly monotonic functions are one-to-one.

## Next Concepts

- **Piecewise Functions** — Inverses of piecewise-defined functions.
- **Implicit Functions** — Functions defined implicitly may have inverses that are also implicit.
- **Higher-Dimensional Inverses** — Inverse functions in multivariable calculus and the Inverse Function Theorem.
- **Normalizing Flows** — Deep generative models built from compositions of invertible transformations.
- **Invertible Neural Networks** — Architectures designed to be fully invertible for memory-efficient training and exact likelihood computation.

## Summary

An inverse function $f^{-1}$ undoes the action of $f$, satisfying $f^{-1}(f(x)) = x$ and $f(f^{-1}(y)) = y$. For an inverse to exist, $f$ must be one-to-one (injective) — passing the horizontal line test. To find the inverse algebraically, swap $x$ and $y$ in $y = f(x)$ and solve for $y$. Functions that are not one-to-one on their natural domain can often be made invertible by restricting the domain. The graph of $f^{-1}$ is the reflection of the graph of $f$ across the line $y = x$. Inverse functions are essential in AI/ML for normalizing flows (invertible generative models), autoencoders (encoding/decoding), change-of-variable density transformations, and solving inverse problems.

## Key Takeaways

- $f^{-1}(f(x)) = x$ and $f(f^{-1}(y)) = y$ — these are the defining equations.
- Only one-to-one (injective) functions have inverses.
- The horizontal line test checks whether a function is one-to-one.
- To find $f^{-1}$: swap $x$ and $y$, then solve for $y$.
- Restrict the domain to make non-injective functions invertible (e.g., $x^2 \to \sqrt{x}$ on $[0, \infty)$).
- The graph of $f^{-1}$ is the reflection of $f$ across $y = x$.
- $\text{dom}(f^{-1}) = \text{range}(f)$ and $\text{range}(f^{-1}) = \text{dom}(f)$.
- $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$ — the order reverses.
- $(f^{-1})'(y) = \frac{1}{f'(x)}$ where $y = f(x)$ and $f'(x) \neq 0$.
- Normalizing flows require invertible functions for exact likelihood computation.
- Autoencoders learn an approximate inverse mapping from latent to data space.
