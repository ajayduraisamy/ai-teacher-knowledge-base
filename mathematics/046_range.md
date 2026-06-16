# Concept: Range

## Concept ID

MATH-046

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define the range of a function as the set of all possible output values.
- Distinguish between codomain and range, and explain why the distinction matters.
- Find the range of a function using algebraic methods, graphical analysis, and reasoning about constraints.
- Determine the range of common function families: linear, quadratic, rational, radical, exponential, logarithmic, and trigonometric.
- Connect range concepts to AI/ML applications including activation function output ranges, valid output spaces for classification vs. regression, and output clipping.

## Prerequisites

- Understanding of functions and function notation (MATH-044).
- Understanding of domain (MATH-045) — the range is closely linked.
- Basic algebra: solving equations, completing the square, solving inequalities.
- Graphing skills: ability to sketch basic function shapes.

## Definition

The **range** (or **image**) of a function $f: A \to B$ is the set of all actual outputs that the function produces. Formally:

$$\text{range}(f) = \{f(x) : x \in A\} = \{y \in B : \exists x \in A \text{ with } f(x) = y\}$$

The range is always a subset of the **codomain** $B$. The codomain is the set that the function is declared to map into, while the range is the set of values that are actually attained.

For example, consider $f: \mathbb{R} \to \mathbb{R}$ defined by $f(x) = x^2$. The codomain is $\mathbb{R}$ (all real numbers), but the range is $[0, \infty)$ because $x^2$ is never negative. The range is a proper subset of the codomain in this case.

If the range equals the codomain, the function is called **surjective** or **onto**.

## Intuition

Think of a function as a machine with an input chute and an output chute. The domain is the set of all objects you can put into the input chute. The range is the set of all objects you have ever seen come out of the output chute (or could possibly come out). The codomain is what the manufacturer says the machine can produce, even if it never produces some of those things.

For example, a vending machine that sells snacks:
- **Domain:** The buttons you can press (A1, A2, B1, B2, ...)
- **Range:** The snacks that actually come out (chips, chocolate bars, cookies)
- **Codomain:** All snacks the machine could theoretically stock (including ones currently sold out)

The range answers the question: "If I try every possible input, what outputs can I get?"

Another analogy: a spotlight shining on a wall. The function is the spotlight, the domain is the set of all possible angles you can point it, and the range is the set of points on the wall that actually get illuminated. The codomain would be the entire wall, but some parts may remain dark.

## Why This Concept Matters

Understanding the range of a function is essential for several reasons:

**1. Equation Solving:** To solve $f(x) = y$, we need to know if $y$ is in the range of $f$. If $y$ is not in the range, the equation has no solution. For example, $x^2 = -1$ has no real solution because $-1$ is not in the range of $f(x) = x^2$.

**2. Inverse Functions:** A function has an inverse only if it is bijective (one-to-one and onto). Even if we restrict to make it one-to-one, the inverse's domain is the original function's range.

**3. Optimization:** When optimizing a function, knowing its range tells us the possible values of the objective. For example, if we know the range of a loss function is $[0, \infty)$, we know the best possible loss is 0.

**4. Model Validation:** In machine learning, the output range of a model must match the problem requirements. A regression model predicting house prices should have range $(0, \infty)$ — negative prices make no sense. A classification model should output probabilities in $[0, 1]$ that sum to 1.

**5. Feature Engineering:** The range of input features affects how models behave. Features with very large ranges (e.g., income in $[0, 10^7]$) can dominate features with small ranges (e.g., number of bedrooms in $[1, 5]$) if not normalized.

**6. Activation Function Selection:** The choice of activation function in a neural network determines the range of neuron outputs, which affects training dynamics. Sigmoid outputs in $(0, 1)$, tanh in $(-1, 1)$, and ReLU in $[0, \infty)$.

## Historical Background

The distinction between codomain and range was clarified during the formalization of set theory in the late 19th and early 20th centuries. Earlier mathematicians, including Euler and Cauchy, worked with functions defined by formulas and implicitly considered the codomain to be $\mathbb{R}$ (or $\mathbb{C}$). The range was simply "all values the formula can take."

The modern distinction became important with the development of abstract algebra and category theory in the 20th century. In category theory, a function (morphism) is defined by its domain, codomain, and the rule — the codomain is an integral part of the function's type signature. This is reflected in programming languages: a function `f: A -> B` has type `B` as its declared return type, even if some values of `B` are never returned.

The term "image" is often used synonymously with "range" in advanced mathematics, though some authors distinguish between "image of a function" (range) and "image of a subset" (the set of outputs from a particular subset of the domain).

## Real World Examples

**Example 1: Temperature Conversion.** The function $C(F) = \frac{5}{9}(F - 32)$ converts Fahrenheit to Celsius. The domain is all real numbers (any Fahrenheit temperature is valid). The range is also all real numbers. Every Celsius temperature corresponds to some Fahrenheit temperature.

**Example 2: Exam Scores.** A function $f(s)$ maps a raw score $s$ (from 0 to 100) to a grade. If $f(s) = \frac{s}{100}$, the range is $[0, 1]$. If $f(s)$ maps to letter grades $\{A, B, C, D, F\}$, the range is a finite set of 5 values.

**Example 3: Shipping Cost.** $C(w) = 10 + 5w$ for $w \in [0, 50]$. The range is $[10, 260]$ dollars. No matter what valid weight you ship, the cost will always be between $10 and $260.

**Example 4: Probability.** A probability function $P(X = x)$ always outputs values in $[0, 1]$. If you compute a probability of 1.5, something has gone wrong — 1.5 is not in the range of any valid probability function.

**Example 5: Profit Function.** $P(x) = 20x - x^2$ for $x \in [0, 20]$. The range is $[0, 100]$. The maximum profit of 100 occurs at $x = 10$ units. Knowing the range helps a business understand potential profit outcomes.

## AI/ML Relevance

Range is a critically important concept in machine learning:

**1. Activation Function Output Ranges.** Every activation function has a characteristic range that affects gradient flow and training:

| Activation | Formula | Range | Properties |
|---|---|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Bounded, causes vanishing gradients for large $|x|$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Bounded, zero-centered (helps training) |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | $[0, \infty)$ | Unbounded above, can cause "dead neurons" |
| Leaky ReLU | $\max(\alpha x, x)$ | $(-\infty, \infty)$ | Unbounded, addresses dead neurons |
| Softmax | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)^K$, sums to 1 | Produces a probability distribution |

**2. Output Layers by Task Type.** The output layer's activation determines the range of model predictions:

- **Regression (unbounded):** Linear activation → range $\mathbb{R}$
- **Regression (positive):** ReLU or softplus → range $[0, \infty)$
- **Regression (bounded):** Sigmoid scaled to $[a, b]$ → range $(a, b)$
- **Binary classification:** Sigmoid → range $(0, 1)$ (probability)
- **Multi-class classification:** Softmax → range $(0, 1)^K$ (probability distribution)
- **Multi-label classification:** Sigmoid per class → range $(0, 1)^K$ (independent probabilities)

**3. Loss Function Ranges.** Different loss functions have different ranges:

- MSE (Mean Squared Error): $[0, \infty)$
- MAE (Mean Absolute Error): $[0, \infty)$
- Cross-entropy loss: $(0, \infty)$
- Binary cross-entropy: $(0, \infty)$
- Hinge loss: $[0, \infty)$

Knowing the range helps interpret loss values. An MSE of 100 might be terrible or acceptable depending on the scale of the target variable.

**4. Output Clipping.** When physical constraints dictate a valid output range, predictions outside that range must be clipped. For example, a model predicting energy consumption should not output negative values. Clipping effectively restricts the model's effective range.

**5. Normalization and Standardization.** Feature scaling transforms the range of input features:
- **Min-max scaling:** $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$ maps to $[0, 1]$
- **Standardization:** $x' = \frac{x - \mu}{\sigma}$ maps to $\mathbb{R}$ (not bounded)
- **Robust scaling:** Uses median and IQR, maps to $\mathbb{R}$

**6. Generative Models.** The range of the latent space in generative models (e.g., VAEs, GANs) is typically $\mathbb{R}^d$. The generated outputs must fall within the valid data range (e.g., pixel values in $[0, 255]$ for images).

**7. Reinforcement Learning.** The policy function $\pi(a|s)$ outputs a probability distribution over actions. The range of each action probability is $[0, 1]$, and the sum over all actions is 1.

## Mathematical Explanation

Finding the range of a function involves determining all possible values of $f(x)$ as $x$ varies over the domain. There are several approaches:

**Method 1: Graphical Analysis.** Sketch the graph and observe the set of $y$-values covered. This is the most intuitive method but may not be precise for complex functions.

**Method 2: Algebraic.** Solve the equation $y = f(x)$ for $x$ in terms of $y$. Then determine the set of $y$ for which $x$ is in the domain of $f$. This is very effective for many functions.

**Method 3: Properties of the Function.** Use knowledge of function behavior:
- Quadratic functions: find vertex to determine min/max
- Trigonometric functions: use known bounds ($\sin x \in [-1, 1]$)
- Exponential functions: $e^x \in (0, \infty)$
- Rational functions: find horizontal asymptotes and critical points

**Method 4: Calculus.** For differentiable functions, find critical points (where $f'(x) = 0$) and evaluate endpoints to determine the range.

**Range of Common Functions:**

| Function Type | Example | Range |
|---|---|---|
| Constant | $f(x) = c$ | $\{c\}$ (single value) |
| Linear (non-constant) | $f(x) = mx + b$, $m \neq 0$ | $\mathbb{R}$ |
| Quadratic (upward) | $f(x) = ax^2 + bx + c$, $a > 0$ | $[f(x_v), \infty)$ where $x_v = -b/(2a)$ |
| Quadratic (downward) | $f(x) = ax^2 + bx + c$, $a < 0$ | $(-\infty, f(x_v)]$ |
| Square root | $f(x) = \sqrt{x}$ | $[0, \infty)$ |
| Exponential | $f(x) = e^x$ | $(0, \infty)$ |
| Logarithmic | $f(x) = \ln x$ | $\mathbb{R}$ |
| Sine | $f(x) = \sin x$ | $[-1, 1]$ |
| Cosine | $f(x) = \cos x$ | $[-1, 1]$ |
| Tangent | $f(x) = \tan x$ | $\mathbb{R}$ |
| Absolute value | $f(x) = |x|$ | $[0, \infty)$ |
| Reciprocal | $f(x) = 1/x$ | $(-\infty, 0) \cup (0, \infty)$ |

**Codomain vs. Range:**

The codomain is declared as part of the function definition. The range is determined by the function itself. For $f: \mathbb{R} \to \mathbb{R}$ with $f(x) = e^x$:
- Codomain = $\mathbb{R}$ (declared)
- Range = $(0, \infty)$ (actual)
- $f$ is not surjective because range $\neq$ codomain

For $g: \mathbb{R} \to (0, \infty)$ with $g(x) = e^x$:
- Codomain = $(0, \infty)$ (declared)
- Range = $(0, \infty)$ (actual)
- $g$ is surjective

## Formula(s)

**Definition of range:**
$$\text{range}(f) = \{y : \exists x \in \text{dom}(f) \text{ with } f(x) = y\}$$

**Range of a quadratic function $f(x) = ax^2 + bx + c$:**
$$\text{vertex: } x_v = -\frac{b}{2a}, \quad y_v = f(x_v) = c - \frac{b^2}{4a}$$
$$\text{range} = \begin{cases} [y_v, \infty), & a > 0 \\ (-\infty, y_v], & a < 0 \end{cases}$$

**Range of $f(x) = a\sin(bx + c) + d$:**
$$[-|a| + d, |a| + d]$$

**Range of $f(x) = ae^{bx} + c$:**
$$\text{range} = \begin{cases} (c, \infty), & a > 0 \\ (-\infty, c), & a < 0 \end{cases}$$

**Range of $f(x) = a \ln(bx + c) + d$ with domain $( -c/b, \infty)$:**
$$\text{range} = \mathbb{R}$$

## Properties

1. **Range $\subseteq$ Codomain.** The range is always a subset (not necessarily proper) of the codomain.

2. **Surjectivity:** A function is surjective (onto) if and only if range = codomain.

3. **Range depends on domain.** Changing the domain changes the range. For $f(x) = x^2$:
   - Domain $\mathbb{R}$: range $[0, \infty)$
   - Domain $[-2, 2]$: range $[0, 4]$
   - Domain $\{1, 2, 3\}$: range $\{1, 4, 9\}$

4. **Range of inverse:** If $f^{-1}$ exists, then $\text{dom}(f^{-1}) = \text{range}(f)$ and $\text{range}(f^{-1}) = \text{dom}(f)$.

5. **Range vs. codomain in practice:** In many practical situations, the codomain is not specified explicitly. We often say "the function $f(x) = \frac{1}{x}$" and assume the natural context of real numbers, where the codomain is $\mathbb{R}$ and the range is $\mathbb{R}\setminus\{0\}$.

6. **Bounded range:** A function with a bounded range has both a finite lower bound and a finite upper bound. For example, $\sin x$ has a bounded range $[-1, 1]$.

7. **Monotonic functions:** If $f$ is strictly monotonic (always increasing or always decreasing) on its domain, then the range can be found by evaluating at domain endpoints.

8. **Continuous functions on intervals:** A continuous function on a closed interval $[a, b]$ has a range that is also a closed interval $[\min f, \max f]$ (by the Extreme Value Theorem).

## Step-by-Step Worked Examples

### Example 1: Range of a Linear Function

Find the range of $f(x) = 3x - 5$ with domain $\mathbb{R}$.

**Step 1:** Recognize that this is a linear function with slope $m = 3 \neq 0$.

**Step 2:** As $x \to -\infty$, $f(x) \to -\infty$. As $x \to \infty$, $f(x) \to \infty$.

**Step 3:** Since linear functions with non-zero slope are continuous and unbounded in both directions, the range is all real numbers.

**Answer:** $\text{range}(f) = \mathbb{R}$

### Example 2: Range of a Quadratic Function

Find the range of $f(x) = 2x^2 - 8x + 7$.

**Step 1:** Identify $a = 2 > 0$, so the parabola opens upward. The range will be $[y_v, \infty)$.

**Step 2:** Find the vertex $x_v = -\frac{b}{2a} = -\frac{-8}{2(2)} = \frac{8}{4} = 2$.

**Step 3:** Evaluate $y_v = f(2) = 2(2)^2 - 8(2) + 7 = 2(4) - 16 + 7 = 8 - 16 + 7 = -1$.

**Step 4:** Since $a > 0$, the minimum value is $-1$, and the function grows without bound as $x \to \pm\infty$.

**Answer:** $\text{range}(f) = [-1, \infty)$

### Example 3: Range of a Rational Function

Find the range of $f(x) = \frac{2x + 1}{x - 3}$.

**Step 1:** Set $y = \frac{2x + 1}{x - 3}$ and solve for $x$ in terms of $y$.
$$y(x - 3) = 2x + 1$$
$$yx - 3y = 2x + 1$$
$$yx - 2x = 3y + 1$$
$$x(y - 2) = 3y + 1$$

**Step 2:** If $y \neq 2$, then $x = \frac{3y + 1}{y - 2}$.

**Step 3:** For this $x$ to be in the domain, we need $x \neq 3$ (but the expression gives some $x$; we also check if the denominator of $y$ is ever zero).

**Step 4:** Check $y = 2$: would $\frac{2x + 1}{x - 3} = 2$ have a solution?
$$2x + 1 = 2(x - 3) = 2x - 6 \implies 1 = -6$$
This is impossible, so $y = 2$ is not in the range.

**Step 5:** For any $y \neq 2$, we found $x = \frac{3y+1}{y-2}$, which is defined and not equal to 3 (since if $x = 3$, then $\frac{3y+1}{y-2} = 3 \implies 3y+1 = 3y-6 \implies 1 = -6$, contradiction).

**Answer:** $\text{range}(f) = (-\infty, 2) \cup (2, \infty)$

### Example 4: Range of a Square Root Function

Find the range of $f(x) = \sqrt{4 - x^2}$.

**Step 1:** The domain is $-2 \leq x \leq 2$ (from $4 - x^2 \geq 0$).

**Step 2:** When $x = \pm 2$, $f(x) = \sqrt{0} = 0$.
When $x = 0$, $f(x) = \sqrt{4} = 2$.

**Step 3:** The function $f(x) = \sqrt{4 - x^2}$ is continuous on $[-2, 2]$. It attains its minimum at the endpoints (0) and its maximum at the center (2).

**Step 4:** Since $4 - x^2$ varies continuously from 0 to 4, the square root varies from 0 to 2, taking every value in between.

**Answer:** $\text{range}(f) = [0, 2]$

### Example 5: Range of an Exponential Function

Find the range of $f(x) = 3e^{2x} + 1$.

**Step 1:** The domain is $\mathbb{R}$.

**Step 2:** $e^{2x} > 0$ for all $x \in \mathbb{R}$. So $3e^{2x} > 0$ for all $x$.

**Step 3:** As $x \to -\infty$, $e^{2x} \to 0^+$, so $f(x) \to 3(0) + 1 = 1^+$ (approaches 1 from above).

**Step 4:** As $x \to \infty$, $e^{2x} \to \infty$, so $f(x) \to \infty$.

**Step 5:** Since $e^{2x}$ is continuous and takes every positive value, $3e^{2x} + 1$ takes every value greater than 1.

**Answer:** $\text{range}(f) = (1, \infty)$

### Example 6: Range of a Trigonometric Function

Find the range of $f(x) = 2\sin(3x - \frac{\pi}{4}) + 1$.

**Step 1:** Recall that $\sin(\theta) \in [-1, 1]$ for any $\theta$.

**Step 2:** The amplitude factor 2 scales the range: $2\sin(\theta) \in [-2, 2]$.

**Step 3:** Adding 1 shifts the range: $2\sin(\theta) + 1 \in [-1, 3]$.

**Step 4:** The horizontal transformations (frequency $3$ and phase shift $-\frac{\pi}{4}$) do not change the range; they only affect where the minimum and maximum occur on the $x$-axis.

**Answer:** $\text{range}(f) = [-1, 3]$

### Example 7: Range of a Function with Restricted Domain

Find the range of $f(x) = x^2 - 2x + 3$ on the domain $[-1, 4]$.

**Step 1:** First find the vertex: $x_v = -\frac{b}{2a} = -\frac{-2}{2(1)} = 1$.

**Step 2:** Since $x_v = 1$ is inside the domain $[-1, 4]$, evaluate $f(1) = 1 - 2 + 3 = 2$.

**Step 3:** Evaluate endpoints: $f(-1) = 1 + 2 + 3 = 6$, $f(4) = 16 - 8 + 3 = 11$.

**Step 4:** Since the parabola opens upward ($a = 1 > 0$), the minimum is at the vertex: $y_{\min} = 2$. The maximum is at an endpoint: $y_{\max} = 11$.

**Step 5:** As a continuous function on a closed interval, the range is $[y_{\min}, y_{\max}]$.

**Answer:** $\text{range}(f) = [2, 11]$

## Visual Interpretation

The range of a function can be visualized on the $y$-axis of the coordinate plane. The range is the set of $y$-coordinates that the graph covers.

**Example 1:** $f(x) = x^2$ has range $[0, \infty)$. The graph is a parabola sitting on the $x$-axis and extending upward. All $y$-values are non-negative.

```
y
|     *
|   *   *
| *       *
|*         *
|----------- x
```

The $y$-axis from 0 upward is covered.

**Example 2:** $f(x) = \sin x$ has range $[-1, 1]$. The wave oscillates between $-1$ and $1$, never going above or below.

```
y 1 ~~~~~*~~~*~~~*~~~*~~~*~~~~~
|       *   *   *   *   *   *
|      * * * * * * * * * * * *
|     *   *   *   *   *   *   *
|-1 ~~~~~~~~~~~~~~~~~~~~~~~~~ x
```

The $y$-axis from $-1$ to $1$ is covered.

**Example 3:** $f(x) = e^x$ has range $(0, \infty)$. The graph approaches but never reaches $y = 0$ (horizontal asymptote).

```
y
|        *
|      *
|    *
|  *
|*
|----|----------------- x
0
```

The $y$-axis above 0 (but not including 0) is covered.

**Mapping Diagram View:**
```
Domain (x)          Range (y)
   1   ──────────►   1
   2   ──────────►   4
   3   ──────────►   9
   4   ──────────►  16
```

Here the domain is $\{1, 2, 3, 4\}$ and the range is $\{1, 4, 9, 16\}$. The codomain might be $\mathbb{R}$, but only these four values are attained.

## Common Mistakes

1. **Confusing range with codomain.** The codomain is declared; the range is what is actually achieved. For $f: \mathbb{R} \to \mathbb{R}$ with $f(x) = e^x$, many students think the range is $\mathbb{R}$ (the codomain), but it is actually $(0, \infty)$.

2. **Assuming the range of $f(x) = \frac{1}{x}$ is $\mathbb{R}$.** The reciprocal function never outputs 0, so the range is $(-\infty, 0) \cup (0, \infty)$.

3. **Forgetting the domain when finding the range.** The range depends on the domain. $f(x) = x^2$ on $[0, 2]$ has range $[0, 4]$, not $[0, \infty)$.

4. **Mistaking the vertex of a quadratic.** For $f(x) = -2x^2 + 4x - 1$, the parabola opens downward ($a = -2 < 0$). Students often give the range as $[y_v, \infty)$ rather than $(-\infty, y_v]$.

5. **Not checking whether the function attains its bounds.** For $f(x) = \frac{1}{x^2 + 1}$, the range is $(0, 1]$. The value 1 is attained at $x = 0$, but 0 is never attained (only approached as $x \to \pm\infty$). So 0 is not included, but 1 is included.

6. **Confusing the range of $\sin x$ and $\cos x$ with their period.** The range of $\sin x$ is $[-1, 1]$, not $[0, 2\pi]$. $[0, 2\pi]$ is one period, not the range.

7. **Thinking that $f(x) = \sqrt{x^2}$ has range $(-\infty, \infty)$.** Actually $\sqrt{x^2} = |x|$, so the range is $[0, \infty)$.

8. **Forgetting to consider the composition of functions changes the range.** For $f(x) = \frac{1}{g(x)}$, the range depends heavily on $g$. If $g(x) = x^2$, the range of $f$ is $(0, \infty)$ (since $x^2 \geq 0$, so $1/x^2 > 0$). But if $g(x) = x^2 - 1$, the range of $f$ is more complex.

## Interview Questions

### Beginner

1. **What is the range of a function? How is it different from the codomain?**
   *Answer: The range is the set of all actual outputs of a function, $\{f(x) : x \in \text{dom}(f)\}$. The codomain is the set the function is defined to map into. The range is always a subset of the codomain. They are equal if and only if the function is surjective.*

2. **Find the range of $f(x) = x^2 + 3$.**
   *Answer: Since $x^2 \geq 0$ for all $x$, we have $x^2 + 3 \geq 3$. The range is $[3, \infty)$.*

3. **Find the range of $f(x) = \sin x$.**
   *Answer: The sine function oscillates between $-1$ and $1$. The range is $[-1, 1]$.*

4. **What is the range of $f(x) = 5$?**
   *Answer: $f(x) = 5$ is a constant function. Every input maps to 5. The range is $\{5\}$, a single value.*

5. **Find the range of $f(x) = \sqrt{x + 2}$.**
   *Answer: The domain is $x \geq -2$. As $x$ varies from $-2$ to $\infty$, $\sqrt{x+2}$ varies from $0$ to $\infty$. The range is $[0, \infty)$.*

6. **Explain why $-1$ is not in the range of $f(x) = |x| + 2$.**
   *Answer: $|x| \geq 0$ for all $x$, so $|x| + 2 \geq 2$. Since $-1 < 2$, it cannot be produced as an output.*

### Intermediate

1. **Find the range of $f(x) = \frac{3}{x^2 + 1}$.**
   *Answer: $x^2 + 1 \geq 1$, so $0 < \frac{3}{x^2+1} \leq 3$. The maximum 3 is attained at $x = 0$. As $x \to \pm\infty$, $f(x) \to 0^+$. The range is $(0, 3]$.*

2. **Find the range of $g(x) = \frac{x^2 - 1}{x^2 + 1}$.**
   *Answer: Write $y = \frac{x^2 - 1}{x^2 + 1}$. Solve for $x^2$: $y(x^2 + 1) = x^2 - 1 \implies yx^2 + y = x^2 - 1 \implies yx^2 - x^2 = -1 - y \implies x^2(y - 1) = -(y + 1) \implies x^2 = \frac{-(y+1)}{y-1} = \frac{y+1}{1-y}$. Since $x^2 \geq 0$, we need $\frac{y+1}{1-y} \geq 0$. Also $y \neq 1$ (denominator). Solving gives $-1 \leq y < 1$. The range is $[-1, 1)$.*

3. **Why does the choice of activation function in the output layer of a neural network depend on the task?**
   *Answer: The output layer's activation function determines the range of predictions, which must match the task. For binary classification, sigmoid gives $(0, 1)$ — interpretable as a probability. For multi-class, softmax gives a probability distribution (sums to 1). For regression, a linear activation (identity) gives $\mathbb{R}$, allowing any real output. For positive-only regression (e.g., price prediction), softplus gives $(0, \infty)$. Choosing the wrong output activation leads to invalid predictions (e.g., negative prices or probabilities outside $[0, 1]$).*

4. **If $f(x) = x^2 - 6x + 10$, find the range of $f$ and the value of $x$ where the minimum occurs.**
   *Answer: Complete the square: $f(x) = (x^2 - 6x + 9) + 1 = (x - 3)^2 + 1$. Since $(x-3)^2 \geq 0$, $f(x) \geq 1$. Minimum $= 1$ at $x = 3$. Range: $[1, \infty)$.*

5. **What is the range of the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$? Why is this range useful for binary classification?**
   *Answer: The range is $(0, 1)$. As $x \to -\infty$, $\sigma(x) \to 0^+$. As $x \to \infty$, $\sigma(x) \to 1^-$. The range $(0, 1)$ is useful because it can be interpreted as a probability of belonging to the positive class. The output never reaches exactly 0 or 1 (only approaches them), which means the model always maintains some uncertainty, which is desirable for numerical stability and to prevent extreme logits from causing issues during training.*

6. **Give an example of a function whose range is a proper subset of its codomain, and explain what this means for invertibility.**
   *Answer: $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$. Codomain = $\mathbb{R}$, range = $[0, \infty)$. Since the range is not the entire codomain, $f$ is not surjective. This means $f$ is not invertible on $\mathbb{R}$ (since, e.g., $-1$ has no preimage). To make $f$ invertible, we must either restrict the codomain to $[0, \infty)$ (making it surjective) or restrict the domain to $[0, \infty)$ (making it injective).*

### Advanced

1. **Find the range of $f(x) = \frac{x^2 - 4}{x - 2}$ (without simplifying).**
   *Answer: The domain is $x \neq 2$. For $x \neq 2$, $\frac{x^2 - 4}{x - 2} = \frac{(x-2)(x+2)}{x-2} = x + 2$. So for all $x \neq 2$, $f(x) = x + 2$, which can take any real value except the value at $x = 2$, which would be $4$. But $f(2)$ is undefined. Therefore the range is $\mathbb{R} \setminus \{4\}$.*

2. **Consider the function $f(x) = \frac{ax + b}{cx + d}$ with $ad - bc \neq 0$. Prove that its range is $\mathbb{R} \setminus \{\frac{a}{c}\}$ (assuming $c \neq 0$).**
   *Proof: Set $y = \frac{ax + b}{cx + d}$. Multiply: $y(cx + d) = ax + b \implies ycx + yd = ax + b \implies ycx - ax = b - yd \implies x(yc - a) = b - yd. If $yc - a \neq 0$, then $x = \frac{b - yd}{yc - a}$. For $x$ to be valid, we need $cx + d \neq 0$ (the original denominator). But the critical case is when $yc - a = 0$, i.e., $y = \frac{a}{c}$. If $y = a/c$, then the equation becomes $0 \cdot x = b - \frac{a}{c}d$, which gives $b - \frac{ad}{c} = \frac{bc - ad}{c} \neq 0$ (since $ad - bc \neq 0$). So there is no solution for $y = a/c$. Therefore the range is all real numbers except $a/c$.*

3. **Explain how the range of the softmax function makes it suitable for multi-class classification, and discuss what happens at extreme temperatures.**
   *Answer: The softmax function $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ maps $\mathbb{R}^K$ to $(0, 1)^K$ with the property that $\sum_i \sigma(\mathbf{z})_i = 1$. This makes the output a valid probability distribution over $K$ classes — the range is exactly the $(K-1)$-dimensional probability simplex. At temperature $T = 1$ (standard), the probabilities reflect the relative magnitudes of logits. As $T \to 0^+$, softmax approaches a one-hot distribution (argmax), with range approaching the vertices of the simplex. As $T \to \infty$, softmax approaches a uniform distribution, with range approaching $\{\frac{1}{K}\}^K$ (a single point). This behavior is used in knowledge distillation (higher $T$ produces softer labels with more information about class similarities) and in reinforcement learning (temperature controls exploration vs. exploitation).*

## Practice Problems

### Easy

1. Find the range of $f(x) = 2x + 5$ on domain $\mathbb{R}$.

2. Find the range of $f(x) = x^2 - 4$.

3. Find the range of $f(x) = -\sqrt{x}$.

4. Find the range of $f(x) = \cos(2x)$.

5. Find the range of $f(x) = 3^x$.

### Medium

1. Find the range of $f(x) = -x^2 + 4x - 1$.

2. Find the range of $f(x) = \frac{1}{x^2 - 4}$.

3. Find the range of $f(x) = 2\sin(x) + 3$.

4. Find the range of $f(x) = \frac{x + 2}{x - 1}$.

5. Find the range of $f(x) = \sqrt{6 - 2x}$ on the domain $[0, 3]$.

### Hard

1. Find the range of $f(x) = \frac{x^2 + 1}{x^2 - 1}$.

2. Find the range of $f(x) = \sqrt{-\ln(\cos x)}$ for $x \in (-\frac{\pi}{2}, \frac{\pi}{2})$.

3. Find the range of $f(x) = \frac{\sin x}{2 + \cos x}$.

## Solutions

### Easy Solutions

**1.** Linear with slope $2 \neq 0$. As $x \to -\infty$, $f(x) \to -\infty$. As $x \to \infty$, $f(x) \to \infty$. Range: $\mathbb{R}$.

**2.** $x^2 \geq 0$, so $x^2 - 4 \geq -4$. Range: $[-4, \infty)$.

**3.** Domain is $x \geq 0$. $\sqrt{x} \geq 0$, so $-\sqrt{x} \leq 0$. As $x \to \infty$, $-\sqrt{x} \to -\infty$. Range: $(-\infty, 0]$.

**4.** $\cos(2x)$ ranges from $-1$ to $1$ as $2x$ varies over $\mathbb{R}$. Range: $[-1, 1]$.

**5.** $3^x > 0$ for all $x \in \mathbb{R}$. As $x \to -\infty$, $3^x \to 0^+$. As $x \to \infty$, $3^x \to \infty$. Range: $(0, \infty)$.

### Medium Solutions

**1.** $a = -1 < 0$, parabola opens downward. Vertex: $x_v = -\frac{b}{2a} = -\frac{4}{2(-1)} = 2$. $f(2) = -4 + 8 - 1 = 3$. Range: $(-\infty, 3]$.

**2.** Domain: $x \neq \pm 2$. As $x \to \pm 2$, $x^2 - 4 \to 0$, so $f(x) \to \pm\infty$ (vertical asymptotes). As $x \to \pm\infty$, $f(x) \to 0^+$ (above the $x$-axis). For $|x| > 2$, $x^2 - 4 > 0$, so $f(x) > 0$. For $|x| < 2$, $-2 < x < 2$, $x^2 - 4 < 0$, so $f(x) < 0$. The minimum of $f$ on $(-2, 2)$ occurs at $x = 0$: $f(0) = -\frac{1}{4}$. So on $(-2, 2)$, $f(x) \in (-\infty, -\frac{1}{4}]$. On $(-\infty, -2) \cup (2, \infty)$, $f(x) \in (0, \infty)$. Range: $(-\infty, -\frac{1}{4}] \cup (0, \infty)$.

**3.** $-1 \leq \sin x \leq 1$, so $2(-1) + 3 \leq 2\sin x + 3 \leq 2(1) + 3$, i.e., $1 \leq f(x) \leq 5$. Range: $[1, 5]$.

**4.** Set $y = \frac{x + 2}{x - 1}$. Solve: $y(x - 1) = x + 2 \implies yx - y = x + 2 \implies yx - x = y + 2 \implies x(y - 1) = y + 2 \implies x = \frac{y + 2}{y - 1}$ (if $y \neq 1$). For $y = 1$: $x + 2 = x - 1 \implies 2 = -1$, impossible. So $y = 1$ is not in the range. Range: $(-\infty, 1) \cup (1, \infty)$.

**5.** Domain $[0, 3]$: when $x = 0$, $f(0) = \sqrt{6} \approx 2.45$. When $x = 3$, $f(3) = \sqrt{0} = 0$. The function is decreasing on $[0, 3]$ (since $6 - 2x$ decreases). Range: $[0, \sqrt{6}]$.

### Hard Solutions

**1.** $y = \frac{x^2 + 1}{x^2 - 1}$. Solve: $y(x^2 - 1) = x^2 + 1 \implies yx^2 - y = x^2 + 1 \implies yx^2 - x^2 = y + 1 \implies x^2(y - 1) = y + 1 \implies x^2 = \frac{y + 1}{y - 1}$. Since $x^2 \geq 0$, we need $\frac{y + 1}{y - 1} \geq 0$. Also $y \neq 1$. Sign analysis: $y \leq -1$ or $y > 1$. But also check if $y = -1$ is attainable: $x^2 = \frac{0}{-2} = 0$, so $x = 0$ gives $f(0) = \frac{1}{-1} = -1$. So $y = -1$ is in the range. As $x \to 1^\pm$, $f(x) \to \pm\infty$. As $x \to \pm\infty$, $f(x) \to 1^+$. Range: $(-\infty, -1] \cup (1, \infty)$.

**2.** For $x \in (-\frac{\pi}{2}, \frac{\pi}{2})$, $\cos x > 0$, so $\ln(\cos x)$ is defined. $\cos x \in (0, 1]$, so $\ln(\cos x) \in (-\infty, 0]$. Therefore $-\ln(\cos x) \in [0, \infty)$. Finally, $\sqrt{-\ln(\cos x)} \in [0, \infty)$. But when does it equal 0? When $-\ln(\cos x) = 0 \implies \ln(\cos x) = 0 \implies \cos x = 1 \implies x = 0$. So $f(0) = 0$. As $x \to \pm\frac{\pi}{2}^-$, $\cos x \to 0^+$, so $\ln(\cos x) \to -\infty$, $-\ln(\cos x) \to \infty$, $f(x) \to \infty$. Range: $[0, \infty)$.

**3.** Set $y = \frac{\sin x}{2 + \cos x}$. Rearranging: $y(2 + \cos x) = \sin x \implies 2y + y\cos x = \sin x \implies \sin x - y\cos x = 2y$. Write $R\sin(x - \phi) = 2y$ where $R = \sqrt{1 + y^2}$ and $\phi = \arctan(y)$. So $\sin(x - \phi) = \frac{2y}{\sqrt{1 + y^2}}$. Since $|\sin| \leq 1$, we require $\left|\frac{2y}{\sqrt{1 + y^2}}\right| \leq 1 \implies \frac{4y^2}{1 + y^2} \leq 1 \implies 4y^2 \leq 1 + y^2 \implies 3y^2 \leq 1 \implies y^2 \leq \frac{1}{3} \implies |y| \leq \frac{1}{\sqrt{3}}$. Range: $[-\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}]$.

## Related Concepts

- **Function** (MATH-044) — The fundamental concept; every function has a range.
- **Domain** (MATH-045) — The set of inputs, closely linked to the range.
- **Codomain** — The set that the function maps into; the range is a subset of the codomain.
- **Composite Function** (MATH-047) — The range of a composition depends on the ranges of the constituent functions.
- **Inverse Function** (MATH-048) — The domain of $f^{-1}$ is the range of $f$.
- **Surjectivity (Onto)** — A function is surjective when its range equals its codomain.
- **Bounded Function** — A function whose range is contained within a finite interval.

## Next Concepts

- **Composite Function** (MATH-047) — Building complex functions from simpler ones, and understanding how ranges compose.
- **Inverse Function** (MATH-048) — Understanding inverses requires a solid grasp of range, since $\text{dom}(f^{-1}) = \text{range}(f)$.
- **Limits at Infinity** — Understanding asymptotic behavior helps determine range endpoints.
- **Extreme Value Theorem** — Guarantees that continuous functions on closed intervals attain their maximum and minimum, determining the range.

## Summary

The range of a function $f$ is the set of all actual output values $\{f(x) : x \in \text{dom}(f)\}$. It is always a subset of the codomain. Finding the range requires understanding the function's behavior: minimum and maximum values (for bounded functions), asymptotes (for rational functions), and the effect of transformations (shifts, stretches, reflections). The range depends critically on the domain: restricting the domain changes the range. In AI/ML, range concepts are essential for choosing activation functions (sigmoid: $(0,1)$, tanh: $(-1,1)$, ReLU: $[0,\infty)$), designing output layers for different tasks, normalizing features, and interpreting model outputs.

## Key Takeaways

- The range is the set of outputs a function actually produces; the codomain is what it could produce.
- The range depends on the domain — changing the domain changes the range.
- For quadratics, find the vertex to determine the range.
- For rational functions, solve $y = f(x)$ for $x$ and determine which $y$ values give valid $x$.
- Continuous functions on closed intervals achieve their min and max (Extreme Value Theorem), giving a closed interval range.
- Activation functions in neural networks have characteristic ranges that affect training dynamics.
- The output layer's activation must have a range matching the task (regression: $\mathbb{R}$, binary classification: $(0,1)$, multi-class: probability simplex).
- The domain of the inverse function is the range of the original function.
