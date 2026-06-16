# Concept: Domain

## Concept ID

MATH-045

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define the domain of a function as the set of all permissible input values.
- Determine the natural domain of a function by identifying restrictions such as division by zero and even roots of negatives.
- Distinguish between natural domain and restricted domain, and explain why restrictions are sometimes imposed.
- Find the domain of sums, differences, products, quotients, and compositions of functions.
- Apply domain concepts in AI/ML contexts, including feature spaces, input constraints, and domain adaptation.

## Prerequisites

- Understanding of what a function is (MATH-044).
- Familiarity with real numbers $\mathbb{R}$ and interval notation.
- Basic algebra: solving linear and quadratic inequalities.
- Knowledge of square roots, absolute values, and rational expressions.

## Definition

The **domain** of a function $f: A \to B$ is the set $A$ of all allowable input values for which $f(x)$ is defined as a real number. If a function is given by an expression without an explicit domain, the **natural domain** (or **maximal domain**) is the largest subset of $\mathbb{R}$ for which the expression yields a real number.

In notation: $\text{dom}(f) = \{x \in \mathbb{R} : f(x) \text{ is defined}\}$.

For example, for $f(x) = \sqrt{x}$, the natural domain is $[0, \infty)$ because the square root of a negative number is not a real number. For $g(x) = \frac{1}{x}$, the natural domain is $(-\infty, 0) \cup (0, \infty)$ because division by zero is undefined.

A **restricted domain** is a subset of the natural domain that we deliberately choose for a specific purpose. For example, we might restrict the domain of $f(x) = x^2$ to $[0, \infty)$ to make it invertible.

## Intuition

Think of the domain as the **guest list** for a function. The function only knows how to handle certain inputs — just as a party only admits certain guests. If you try to feed the function an input that is not on the guest list, the function cannot process it.

For example, consider a coffee vending machine that accepts coins. If you insert a coin (input), the machine functions. But if you insert a paper bill or a foreign coin, the machine rejects it — that input is not in its domain.

In practical terms, the domain answers the question: "What values of $x$ can I plug into this function without breaking math?" Breaking math includes:
- Dividing by zero (mathematically undefined)
- Taking the square root (or any even root) of a negative number (not real)
- Taking the logarithm of a non-positive number (undefined)
- Taking the tangent of $\frac{\pi}{2} + n\pi$ (undefined)

## Why This Concept Matters

The domain of a function is the first thing you must check before doing any work. If you evaluate a function at a point outside its domain, you get an undefined result. This has serious consequences:

- **In engineering and physics:** An incorrect domain can lead to physically impossible solutions (e.g., negative time, infinite temperature).
- **In economics:** A demand function $D(p)$ might only be valid for prices $p \in [0, p_{\text{max}}]$. Extrapolating beyond this domain gives meaningless results.
- **In computer science:** A program that receives an input outside its expected domain may crash, produce garbage output, or create a security vulnerability.
- **In machine learning:** The domain of a model's input features defines the space of possible data points the model can process. Features outside the training domain often lead to unreliable predictions (extrapolation failure).

Understanding domain is essential for safely applying mathematical models to real-world problems. It prevents us from using formulas in situations where they are not valid.

## Historical Background

The concept of domain evolved alongside the formalization of functions. Early mathematicians like Euler and the Bernoullis worked with functions defined by formulas and implicitly assumed the domain was wherever the formula made sense. The notion of explicitly specifying the domain was part of the 19th-century rigorization of analysis by Dirichlet, Riemann, and Weierstrass.

Dirichlet's definition of a function (1837) specified that a function is a rule that assigns a value to each element of a given set — this made the domain an essential part of the definition rather than an afterthought.

The concept of "natural domain" gained importance in complex analysis, where functions like $\sqrt{z}$ require careful handling of branch cuts — subsets of the complex plane where the function is not defined or is multi-valued.

In modern mathematics, the domain is considered an integral part of the function definition. Two functions with different domains are considered different functions even if they agree where both are defined.

## Real World Examples

**Example 1: Shipping Cost.** A shipping company charges $C(w) = 10 + 3w$ dollars for packages weighing $w$ kilograms, but only accepts packages from $0.1$ kg to $50$ kg. The domain is $[0.1, 50]$. A package weighing $100$ kg or $-5$ kg is outside the domain and cannot be shipped.

**Example 2: Blood Alcohol Concentration.** A model predicting impairment based on BAC (blood alcohol concentration) is only valid for BAC $\geq 0\%$. A negative BAC is physically impossible. The domain is $[0, \infty)$, though practically it is $[0, 0.5]$ (above which a person would be comatose or dead).

**Example 3: Height vs. Weight.** A linear regression model predicts weight from height using data from adult humans. The model's domain of validity is roughly $[1.2, 2.2]$ meters (the range of adult human heights). Inputting a height of $0.5$ meters (a baby) or $5$ meters (a giraffe) would give nonsensical predictions.

**Example 4: Speed of a Car.** The function $d(t) = 60t$ gives distance travelled in $t$ hours at 60 km/h. The domain is $t \geq 0$ (we cannot travel for negative time). If we restrict to a 3-hour trip, the domain becomes $[0, 3]$.

**Example 5: Website Traffic.** The number of visitors to a website as a function of time of day $t$ (in hours from midnight) has domain $[0, 24)$. Outside this domain, the input doesn't correspond to a valid time.

## AI/ML Relevance

Domain is a critical concept throughout machine learning:

**1. Feature Space (Input Domain).** In supervised learning, the feature space $\mathcal{X}$ is the domain of the model. For tabular data with $d$ features, $\mathcal{X} \subseteq \mathbb{R}^d$. Each feature has its own domain: age might be $[0, 120]$, income might be $[0, \infty)$, a binary feature might be $\{0, 1\}$. Understanding the domain of each feature is essential for data preprocessing, normalization, and model selection.

**2. Domain Adaptation.** Domain adaptation is a subfield of transfer learning where the training data (source domain) and test data (target domain) come from different but related distributions. A model trained on photographs of objects (source domain) might perform poorly on sketches of the same objects (target domain) because the input domains differ. Domain adaptation techniques try to learn features that are invariant across domains.

**3. Out-of-Distribution (OOD) Detection.** When a deployed ML model receives an input that falls outside the domain of its training data, the model's prediction is unreliable. OOD detection is the task of identifying such inputs so they can be flagged for human review or rejected. For example, a self-driving car's perception system trained on sunny weather data has a domain limited to those conditions — rain or snow are out-of-distribution.

**4. Extrapolation Failure.** Most ML models (especially neural networks) are excellent at interpolation (predicting within the training domain) but poor at extrapolation (predicting outside it). A linear regression model trained on house sizes $[50, 200] \text{ m}^2$ might predict a negative price for a $0 \text{ m}^2$ house — a nonsensical result from being forced to predict outside its domain.

**5. Input Constraints in Optimization.** In constrained optimization (e.g., training a model with L2 regularization), the domain of the weight space is $\mathbb{R}^d$. But with L1 regularization, we are effectively restricting the domain to be a diamond-shaped region (the $\ell^1$ ball). Understanding these domain restrictions is key to understanding regularization.

**6. Data Augmentation.** Data augmentation techniques (rotation, cropping, color jitter for images) effectively enlarge the training domain by creating new valid inputs within a neighborhood of existing data points. This helps the model generalize better within a broader domain.

**7. Activation Function Domains.** Every activation function has a natural domain:
- ReLU: $(-\infty, \infty)$
- Sigmoid: $(-\infty, \infty)$
- Softmax: $\mathbb{R}^K$ (vector input)
- Tanh: $(-\infty, \infty)$

While all common activation functions accept any real input, their outputs have different ranges, which affects the domain of subsequent layers.

## Mathematical Explanation

Finding the domain of a function involves identifying all real numbers $x$ for which the expression is defined. Here are the key restrictions:

**1. Rational Functions (Denominators).** For $f(x) = \frac{P(x)}{Q(x)}$, the domain excludes values where $Q(x) = 0$.

Example: $f(x) = \frac{1}{x - 3}$. The denominator is zero when $x = 3$, so $\text{dom}(f) = (-\infty, 3) \cup (3, \infty)$.

**2. Even Roots.** For $f(x) = \sqrt[n]{g(x)}$ where $n$ is even, we require $g(x) \geq 0$.

Example: $f(x) = \sqrt{4 - x^2}$. We need $4 - x^2 \geq 0$, so $x^2 \leq 4$, giving $\text{dom}(f) = [-2, 2]$.

**3. Odd Roots.** For $f(x) = \sqrt[n]{g(x)}$ where $n$ is odd, the domain is all $x$ where $g(x)$ is defined (odd roots of negative numbers are real). No additional restriction.

Example: $f(x) = \sqrt[3]{x}$ has domain $\mathbb{R}$.

**4. Logarithms.** For $f(x) = \log_a(g(x))$, we require $g(x) > 0$ (and $a > 0$, $a \neq 1$).

Example: $f(x) = \ln(x^2 - 1)$. We need $x^2 - 1 > 0$, so $x < -1$ or $x > 1$. Domain: $(-\infty, -1) \cup (1, \infty)$.

**5. Combinations.** When combining functions through addition, subtraction, multiplication, division, or composition, the domain is the intersection of the individual domains (with additional restrictions for division and composition).

For $f(x) = \sqrt{x-1} + \frac{1}{x-3}$:
- $\sqrt{x-1}$ requires $x \geq 1$
- $\frac{1}{x-3}$ requires $x \neq 3$
- Intersection: $[1, 3) \cup (3, \infty)$

**6. Piecewise Functions.** The domain is the union of the domains of each piece.

Example: $f(x) = \begin{cases} x^2, & x < 0 \\ \sqrt{x}, & x \geq 0 \end{cases}$ has domain $\mathbb{R}$ because all $x$ are covered.

**Interval Notation:**
- $(a, b)$: $a < x < b$ (open interval)
- $[a, b]$: $a \leq x \leq b$ (closed interval)
- $(a, b]$: $a < x \leq b$ (half-open)
- $[a, \infty)$: $x \geq a$
- $(-\infty, b)$: $x < b$
- $\mathbb{R}$ or $(-\infty, \infty)$: all real numbers

## Formula(s)

**Domain of a rational function:**
$$\text{dom}\left(\frac{P(x)}{Q(x)}\right) = \{x \in \mathbb{R} : Q(x) \neq 0\}$$

**Domain of a function with an even root:**
$$\text{dom}\left(\sqrt[2n]{g(x)}\right) = \{x \in \mathbb{R} : g(x) \geq 0\}$$

**Domain of a logarithmic function:**
$$\text{dom}\left(\log_a(g(x))\right) = \{x \in \mathbb{R} : g(x) > 0\}$$

**Domain of a composite function $(f \circ g)(x) = f(g(x))$:**
$$\text{dom}(f \circ g) = \{x \in \text{dom}(g) : g(x) \in \text{dom}(f)\}$$

**Domain of sum/difference/product:**
$$\text{dom}(f \pm g) = \text{dom}(f \cdot g) = \text{dom}(f) \cap \text{dom}(g)$$

**Domain of quotient $\frac{f}{g}$:**
$$\text{dom}\left(\frac{f}{g}\right) = (\text{dom}(f) \cap \text{dom}(g)) \setminus \{x : g(x) = 0\}$$

## Properties

1. **Subset of $\mathbb{R}$:** For real-valued functions of a real variable, the domain is a subset of $\mathbb{R}$ (often an interval or union of intervals).

2. **Natural vs. Restricted:** The natural domain is the maximal set where the function is defined. A restricted domain is a chosen subset of the natural domain.

3. **Context-dependent:** The same expression can have different domains in different contexts. For example, $\sqrt{x}$ has domain $[0, \infty)$ in real analysis but $\mathbb{C}$ in complex analysis.

4. **Domain intersection for combinations:** When combining functions, the domain of the result is the intersection (overlap) of the individual domains, possibly with additional restrictions.

5. **Empty domain:** If no $x$ satisfies all restrictions, the domain is the empty set $\emptyset$, and the function is nowhere defined. For example, $f(x) = \frac{1}{\sqrt{-x^2 - 1}}$ has domain $\emptyset$.

6. **Domain of inverse:** If $f$ has an inverse $f^{-1}$, then $\text{dom}(f^{-1}) = \text{range}(f)$ and $\text{range}(f^{-1}) = \text{dom}(f)$.

7. **Closed under the function:** By definition, if $x$ is in the domain, $f(x)$ is defined and yields a real number.

## Step-by-Step Worked Examples

### Example 1: Domain of a Rational Function

Find the domain of $f(x) = \frac{2x + 5}{x^2 - 4x + 3}$.

**Step 1:** Identify the restriction. The denominator cannot be zero.
$$x^2 - 4x + 3 \neq 0$$

**Step 2:** Solve the quadratic equation $x^2 - 4x + 3 = 0$.
$$(x - 1)(x - 3) = 0$$
$$x = 1 \quad \text{or} \quad x = 3$$

**Step 3:** The domain is all real numbers except $x = 1$ and $x = 3$.

**Answer:** $\text{dom}(f) = (-\infty, 1) \cup (1, 3) \cup (3, \infty)$

### Example 2: Domain with a Square Root

Find the domain of $f(x) = \sqrt{9 - x^2}$.

**Step 1:** The expression under the square root must be non-negative.
$$9 - x^2 \geq 0$$

**Step 2:** Solve the inequality.
$$x^2 \leq 9$$
$$|x| \leq 3$$
$$-3 \leq x \leq 3$$

**Answer:** $\text{dom}(f) = [-3, 3]$

### Example 3: Domain with Both Square Root and Denominator

Find the domain of $f(x) = \frac{\sqrt{x + 2}}{x - 5}$.

**Step 1:** Square root restriction: $x + 2 \geq 0$, so $x \geq -2$.

**Step 2:** Denominator restriction: $x - 5 \neq 0$, so $x \neq 5$.

**Step 3:** Combine restrictions. The domain is $x \geq -2$ and $x \neq 5$.

**Answer:** $\text{dom}(f) = [-2, 5) \cup (5, \infty)$

### Example 4: Domain of a Logarithmic Function

Find the domain of $f(x) = \log_3(2x - 4)$.

**Step 1:** The argument of the logarithm must be positive.
$$2x - 4 > 0$$

**Step 2:** Solve the inequality.
$$2x > 4$$
$$x > 2$$

**Answer:** $\text{dom}(f) = (2, \infty)$

### Example 5: Domain with a Fourth Root

Find the domain of $f(x) = \frac{1}{\sqrt[4]{5 - x}}$.

**Step 1:** The fourth root (even index) requires its argument to be non-negative.
$$5 - x \geq 0 \implies x \leq 5$$

**Step 2:** However, the fourth root is in the denominator, so it cannot be zero.
$$\sqrt[4]{5 - x} \neq 0 \implies 5 - x \neq 0 \implies x \neq 5$$

**Step 3:** Combine: $x < 5$.

**Answer:** $\text{dom}(f) = (-\infty, 5)$

### Example 6: Domain of a Combined Function

Find the domain of $f(x) = \frac{\sqrt{x - 1}}{x - 2} + \ln(6 - x)$.

**Step 1:** Square root restriction: $x - 1 \geq 0 \implies x \geq 1$.

**Step 2:** Denominator restriction (first term): $x - 2 \neq 0 \implies x \neq 2$.

**Step 3:** Logarithm restriction: $6 - x > 0 \implies x < 6$.

**Step 4:** Combine all restrictions: $x \geq 1$, $x \neq 2$, $x < 6$.

**Answer:** $\text{dom}(f) = [1, 2) \cup (2, 6)$

### Example 7: Domain with an Absolute Value

Find the domain of $f(x) = \frac{1}{|x| - 3}$.

**Step 1:** Denominator cannot be zero.
$$|x| - 3 \neq 0 \implies |x| \neq 3 \implies x \neq \pm 3$$

**Step 2:** No other restrictions (absolute value is defined for all real numbers).

**Answer:** $\text{dom}(f) = (-\infty, -3) \cup (-3, 3) \cup (3, \infty)$

### Example 8: Domain of a Piecewise Function

Find the domain of $f(x) = \begin{cases} \frac{1}{x}, & x < 0 \\ \sqrt{x}, & x \geq 0 \end{cases}$.

**Step 1:** First piece: $\frac{1}{x}$ with $x < 0$. The denominator restriction $x \neq 0$ is automatically satisfied since $x < 0$. So the first piece contributes $(-\infty, 0)$.

**Step 2:** Second piece: $\sqrt{x}$ with $x \geq 0$. The square root requires $x \geq 0$, which matches the condition. So the second piece contributes $[0, \infty)$.

**Step 3:** Union: $(-\infty, 0) \cup [0, \infty) = \mathbb{R}$.

**Answer:** $\text{dom}(f) = \mathbb{R}$

## Visual Interpretation

The domain of a function can be visualized on the $x$-axis of the coordinate plane. The domain is the set of $x$-coordinates for which the graph exists.

**Example 1:** $f(x) = \sqrt{x}$ has domain $[0, \infty)$. The graph only exists to the right of (and including) $x = 0$.

```
y
|     *
|   *
| *
|*
|----|----------------- x
0
```

**Example 2:** $f(x) = \frac{1}{x}$ has domain $(-\infty, 0) \cup (0, \infty)$. The graph has a vertical asymptote at $x = 0$, meaning the function is not defined there.

```
y
|  \    /
|   \  /
|    \/
|----|----x----|---- x
|    /\   0
|   /  \
|  /    \
```

**Example 3:** $f(x) = \sqrt{9 - x^2}$ has domain $[-3, 3]$. The graph is a semicircle (top half of $x^2 + y^2 = 9$). It only exists between $x = -3$ and $x = 3$.

```
y
|    ___
|  /     \
| /       \
|/         \
|----|----|---- x
   -3   0    3
```

**Number Line Representation:** For $f(x) = \frac{\sqrt{x+2}}{x-5}$ with domain $[-2, 5) \cup (5, \infty)$:

```
<───|═══════|═══════○═══════════════|───>
   -2       5       (open circle)
```

Where $=$ indicates the domain, $\bullet$ indicates inclusion, and $\circ$ indicates exclusion.

## Common Mistakes

1. **Forgetting to exclude values that make a denominator zero.** This is the most common domain error. Given $f(x) = \frac{1}{x-2}$, students often say the domain is $\mathbb{R}$, forgetting to exclude $x = 2$.

2. **Assuming $\sqrt{x^2} = x$ and ignoring domain consequences.** Actually, $\sqrt{x^2} = |x|$, not $x$. The function $f(x) = \sqrt{x^2}$ has domain $\mathbb{R}$ (correct), but simplifying to $x$ changes the function (since $x$ can be negative while $\sqrt{x^2}$ is always non-negative).

3. **Confusing the domain of $f(g(x))$ with the domain of $f$ or $g$ alone.** For $f(x) = \frac{1}{x}$ and $g(x) = x - 1$, the composite $f(g(x)) = \frac{1}{x-1}$ has domain $(-\infty, 1) \cup (1, \infty)$, not $\mathbb{R}\setminus\{0\}$ (which is the domain of $f$ alone).

4. **Ignoring the domain when cancelling factors in rational functions.** In $f(x) = \frac{(x-1)(x+2)}{x-1}$, cancelling gives $g(x) = x+2$, but $f$ has domain $\mathbb{R}\setminus\{1\}$ while $g$ has domain $\mathbb{R}$. They are different functions.

5. **Thinking the domain of $\sqrt{x}$ includes negative numbers when complex numbers are allowed.** In real analysis, the domain of $\sqrt{x}$ is $[0, \infty)$. In complex analysis, it is $\mathbb{C}$ (but then $\sqrt{x}$ is multi-valued). Always specify whether working in $\mathbb{R}$ or $\mathbb{C}$.

6. **Forgetting to consider the domain of trigonometric functions.** $\tan x = \frac{\sin x}{\cos x}$ has domain excluding $x = \frac{\pi}{2} + n\pi$. $\sec x = \frac{1}{\cos x}$ excludes the same values.

7. **Assuming piecewise function domains cover all real numbers without checking.** A piecewise function might have gaps. Always verify that the union of all pieces covers the intended domain.

## Interview Questions

### Beginner

1. **What is the domain of a function?**
   *Answer: The domain of a function $f$ is the set of all allowable input values for which $f(x)$ is defined. For a real-valued function, it is the set of $x$-values that produce a real output.*

2. **Find the domain of $f(x) = \frac{1}{x-4}$.**
   *Answer: The denominator cannot be zero, so $x \neq 4$. Domain: $(-\infty, 4) \cup (4, \infty)$.*

3. **Find the domain of $f(x) = \sqrt{x-3}$.**
   *Answer: The expression under the square root must be non-negative: $x-3 \geq 0$, so $x \geq 3$. Domain: $[3, \infty)$.*

4. **Find the domain of $f(x) = \frac{1}{\sqrt{x}}$.**
   *Answer: Two restrictions: $\sqrt{x} \neq 0$ (denominator) and $x \geq 0$ (square root). The square root is zero when $x=0$. So $x > 0$. Domain: $(0, \infty)$.*

5. **What is the domain of $f(x) = x^2 + 3x - 5$?**
   *Answer: There are no denominators, even roots, or logarithms. The function is a polynomial, which is defined for all real numbers. Domain: $\mathbb{R}$ or $(-\infty, \infty)$.*

6. **Explain the difference between natural domain and restricted domain.**
   *Answer: The natural domain is the largest set of inputs for which the function is mathematically defined. A restricted domain is a subset of the natural domain that we deliberately choose, often to make the function invertible or to model a specific real-world constraint.*

### Intermediate

1. **Find the domain of $f(x) = \sqrt{x^2 - 5x + 6}$.**
   *Answer: We need $x^2 - 5x + 6 \geq 0$. Factor: $(x-2)(x-3) \geq 0$. Sign analysis: the quadratic is non-negative outside the interval $[2, 3]$. Domain: $(-\infty, 2] \cup [3, \infty)$.*

2. **Find the domain of $f(x) = \ln(x^2 - 4) + \frac{1}{x-1}$.**
   *Answer: Logarithm: $x^2 - 4 > 0 \implies x < -2$ or $x > 2$ (i.e., $|x| > 2$). Denominator: $x \neq 1$. Intersection: $|x| > 2$ already excludes $x = 1$. Domain: $(-\infty, -2) \cup (2, \infty)$.*

3. **If $f(x) = \sqrt{x}$ and $g(x) = x^2 - 1$, find the domain of $(f \circ g)(x)$.**
   *Answer: $(f \circ g)(x) = f(g(x)) = \sqrt{x^2 - 1}$. Need $x^2 - 1 \geq 0$, so $|x| \geq 1$. Domain: $(-\infty, -1] \cup [1, \infty)$. Note: $g$ has domain $\mathbb{R}$, but $f$ restricts the composite.*

4. **Why is understanding the domain of input features important in machine learning?**
   *Answer: Each feature in a dataset has a natural domain (e.g., age in $[0, 120]$, income in $[0, \infty)$). If a model receives a feature value outside the domain seen in training (e.g., age 200), the model will extrapolate unreliably. Domain knowledge helps with data validation, feature engineering, and choosing appropriate models. Also, many models implicitly assume features come from a specific domain (e.g., $\mathbb{R}^n$ for linear regression, $[0,1]^n$ for models using sigmoid inputs).*

5. **Find the domain of $f(x) = \frac{\sqrt{4 - x^2}}{\ln(x + 1)}$.**
   *Answer: Numerator: $4 - x^2 \geq 0 \implies x \in [-2, 2]$. Denominator: $\ln(x+1) \neq 0$ and $x+1 > 0$. The logarithm is zero when $x+1 = 1 \implies x = 0$. Also $x+1 > 0 \implies x > -1$. Intersection of $[-2, 2]$, $(-1, \infty)$, and $x \neq 0$: $(-1, 0) \cup (0, 2]$.*

6. **What is domain adaptation in machine learning? Give a concrete example.**
   *Answer: Domain adaptation is the process of adapting a model trained on one domain (source) to perform well on a different but related domain (target). Example: A sentiment analysis model trained on product reviews (source domain) is adapted to work on social media posts (target domain). The features (words, phrases) may differ between domains, but the underlying task (sentiment classification) is the same. Techniques include fine-tuning, adversarial domain adaptation, and feature alignment.*

### Advanced

1. **Consider $f(x) = \sqrt{\sin x - \frac{1}{2}}$. Find its domain on $[0, 2\pi]$.**
   *Answer: We need $\sin x - \frac{1}{2} \geq 0$, i.e., $\sin x \geq \frac{1}{2}$. On $[0, 2\pi]$, $\sin x = \frac{1}{2}$ at $x = \frac{\pi}{6}$ and $x = \frac{5\pi}{6}$. $\sin x \geq \frac{1}{2}$ for $x \in [\frac{\pi}{6}, \frac{5\pi}{6}]$. Domain: $[\frac{\pi}{6}, \frac{5\pi}{6}]$.*

2. **Define the concept of "domain of a neural network" formally. How does it relate to the concept of "out-of-distribution" detection?**
   *Answer: The domain of a neural network $f_\theta: \mathcal{X} \to \mathcal{Y}$ is the input space $\mathcal{X} \subseteq \mathbb{R}^d$. In practice, the effective domain is the support of the training distribution $\text{supp}(p_{\text{train}}) \subseteq \mathcal{X}$ because the network is only guaranteed to perform well on inputs similar to the training data. Out-of-distribution (OOD) detection is the task of determining whether a given input $x$ lies in $\text{supp}(p_{\text{train}})$ (or more precisely, whether $p_{\text{train}}(x) > \delta$ for some threshold $\delta$). If $x$ is OOD, the network's output is unreliable. Formal approaches include density estimation, energy-based models, and confidence scoring. The key insight is that the function $f_\theta$ is only well-defined (in a practical sense) on its training domain, even though it may accept inputs from the entire mathematical domain $\mathbb{R}^d$.*

3. **In complex analysis, the function $f(z) = \sqrt{z}$ is multi-valued and requires a branch cut to become a well-defined function. Explain how this relates to the concept of domain restriction.**
   *Answer: In complex analysis, $\sqrt{z} = r^{1/2}e^{i\theta/2}$ where $z = re^{i\theta}$. Because $\theta$ can be increased by $2\pi$ without changing $z$, $\sqrt{z}$ has two possible values (differing by a sign). To make $\sqrt{z}$ a single-valued function, we must restrict the domain by choosing a branch cut — a curve in the complex plane across which the function is discontinuous. The standard choice is to cut along the negative real axis: $\text{dom}(\sqrt{z}) = \mathbb{C} \setminus (-\infty, 0]$, with $\theta \in (-\pi, \pi]$. This is analogous to restricting the domain of $f(x) = \sqrt{x}$ in real analysis to $[0, \infty)$. Both are domain restrictions to ensure single-valuedness. In ML, complex-valued neural networks must carefully handle such cuts.*

## Practice Problems

### Easy

1. Find the domain of $f(x) = \frac{3}{x + 2}$.

2. Find the domain of $f(x) = \sqrt{2x - 6}$.

3. Find the domain of $f(x) = \ln(x - 5)$.

4. Find the domain of $f(x) = \frac{2x}{x^2 - 9}$.

5. Find the domain of $f(x) = \sqrt[4]{x + 1}$.

### Medium

1. Find the domain of $f(x) = \frac{\sqrt{x - 2}}{x^2 - 16}$.

2. Find the domain of $f(x) = \ln(3 - 2x) + \frac{1}{x}$.

3. Find the domain of $f(x) = \frac{x + 1}{\sqrt{x^2 - 4}}$.

4. Given $f(x) = \frac{1}{x-1}$ and $g(x) = \sqrt{x+3}$, find the domain of $(f \circ g)(x)$.

5. Find the domain of $f(x) = \sqrt{5 - |x|}$.

### Hard

1. Find the domain of $f(x) = \sqrt{\frac{x-2}{x+3}}$.

2. Find the domain of $f(x) = \frac{1}{\sqrt{\sin x}} + \ln(\cos x)$ for $x \in [0, 2\pi]$.

3. Determine all values of $a$ for which the domain of $f(x) = \sqrt{ax^2 + 2ax + 3}$ is $\mathbb{R}$.

## Solutions

### Easy Solutions

**1.** Denominator: $x + 2 \neq 0 \implies x \neq -2$. Domain: $(-\infty, -2) \cup (-2, \infty)$.

**2.** Radicand: $2x - 6 \geq 0 \implies x \geq 3$. Domain: $[3, \infty)$.

**3.** Argument: $x - 5 > 0 \implies x > 5$. Domain: $(5, \infty)$.

**4.** Denominator: $x^2 - 9 \neq 0 \implies (x-3)(x+3) \neq 0 \implies x \neq \pm 3$. Domain: $(-\infty, -3) \cup (-3, 3) \cup (3, \infty)$.

**5.** Even root: $x + 1 \geq 0 \implies x \geq -1$. Domain: $[-1, \infty)$.

### Medium Solutions

**1.** Square root: $x - 2 \geq 0 \implies x \geq 2$. Denominator: $x^2 - 16 \neq 0 \implies x \neq \pm 4$. Since $x \geq 2$, the $x = -4$ restriction is irrelevant. So domain: $[2, 4) \cup (4, \infty)$.

**2.** Logarithm: $3 - 2x > 0 \implies x < 1.5$. Denominator (second term): $x \neq 0$. Domain: $(-\infty, 0) \cup (0, 1.5)$.

**3.** Denominator: $\sqrt{x^2 - 4} \neq 0$, so $x^2 - 4 \neq 0 \implies x \neq \pm 2$. Also square root requires $x^2 - 4 \geq 0 \implies |x| \geq 2$. Combining: $|x| > 2$. Domain: $(-\infty, -2) \cup (2, \infty)$.

**4.** $(f \circ g)(x) = f(g(x)) = \frac{1}{\sqrt{x+3} - 1}$. Domain of $g$: $x + 3 \geq 0 \implies x \geq -3$. Also, denominator of $f \circ g$: $\sqrt{x+3} - 1 \neq 0 \implies \sqrt{x+3} \neq 1 \implies x+3 \neq 1 \implies x \neq -2$. Domain: $[-3, -2) \cup (-2, \infty)$.

**5.** Need $5 - |x| \geq 0 \implies |x| \leq 5 \implies -5 \leq x \leq 5$. Domain: $[-5, 5]$.

### Hard Solutions

**1.** The expression under the square root must be non-negative: $\frac{x-2}{x+3} \geq 0$. Also, the denominator $x+3$ cannot be zero when evaluating the fraction itself, but since it's under a square root we only need the overall expression to be non-negative.

We solve $\frac{x-2}{x+3} \geq 0$:
- Numerator zero: $x = 2$
- Denominator zero: $x = -3$ (not in domain)
- Sign analysis:
  - $x < -3$: numerator negative, denominator negative → fraction positive ✓
  - $-3 < x < 2$: numerator negative, denominator positive → fraction negative ✗
  - $x > 2$: numerator positive, denominator positive → fraction positive ✓
- Include $x = 2$ (fraction = 0, square root of 0 is defined).
- Exclude $x = -3$ (division by zero).

Domain: $(-\infty, -3) \cup [2, \infty)$

**2.** For $\frac{1}{\sqrt{\sin x}}$: $\sin x > 0$ (since $\sin x = 0$ makes denominator zero). So $\sin x > 0 \implies x \in (0, \pi)$.

For $\ln(\cos x)$: $\cos x > 0 \implies x \in (-\frac{\pi}{2}, \frac{\pi}{2})$. On $[0, 2\pi]$, this gives $[0, \frac{\pi}{2}) \cup (\frac{3\pi}{2}, 2\pi]$.

Intersection of $\sin x > 0$ (i.e., $(0, \pi)$) and $\cos x > 0$ (i.e., $[0, \frac{\pi}{2}) \cup (\frac{3\pi}{2}, 2\pi]$):
- $(0, \pi) \cap [0, \frac{\pi}{2}) = (0, \frac{\pi}{2})$
- $(0, \pi) \cap (\frac{3\pi}{2}, 2\pi] = \emptyset$

Also check $x = 0$: $\sin 0 = 0$ makes denominator zero. So $0$ is excluded.

Domain: $(0, \frac{\pi}{2})$

**3.** For the domain to be $\mathbb{R}$, we need $ax^2 + 2ax + 3 \geq 0$ for all $x \in \mathbb{R}$.

Case 1: $a = 0$. Then $f(x) = \sqrt{3}$, which has domain $\mathbb{R}$. ✓

Case 2: $a > 0$. The quadratic opens upward. For it to be non-negative for all $x$, its discriminant must be $\leq 0$.
$$(2a)^2 - 4(a)(3) \leq 0 \implies 4a^2 - 12a \leq 0 \implies 4a(a - 3) \leq 0 \implies 0 \leq a \leq 3$$
Since $a > 0$, we have $0 < a \leq 3$.

Case 3: $a < 0$. The quadratic opens downward. It will be negative for some $x$ (large $|x|$), so the domain cannot be $\mathbb{R}$.

Combining: $a \in [0, 3]$.

## Related Concepts

- **Function** (MATH-044) — A function is a mapping from a domain to a codomain. The domain is an integral part of the function definition.
- **Range** (MATH-046) — The set of all outputs produced by a function, closely related to the domain.
- **Composite Function** (MATH-047) — The domain of a composite $f(g(x))$ requires $x$ to be in the domain of $g$ and $g(x)$ to be in the domain of $f$.
- **Inverse Function** (MATH-048) — The domain of $f^{-1}$ is the range of $f$, and vice versa.
- **Continuity** — A function is continuous on its domain if it has no breaks, jumps, or holes.
- **Interval Notation** — The standard way to express domains of real functions.

## Next Concepts

- **Range** (MATH-046) — Understanding how to find the set of all possible outputs, complementing domain knowledge.
- **Composite Function** (MATH-047) — Building and analyzing complex functions, including domain considerations for compositions.
- **Inverse Function** (MATH-048) — Functions that reverse a mapping, with domain and range swapped.
- **Limits** — The foundation of calculus, which studies function behavior as inputs approach domain boundaries.

## Summary

The domain of a function is the set of all permissible input values. For real-valued functions, the natural domain is determined by restrictions that would make the expression undefined: division by zero, even roots of negative numbers, logarithms of non-positive numbers, and certain trigonometric values. When combining functions through arithmetic or composition, the domain of the result is the intersection of the individual domains with additional restrictions. Domain is critical in AI/ML for understanding feature spaces, out-of-distribution detection, domain adaptation, and ensuring reliable model predictions.

## Key Takeaways

- The domain is the set of all inputs for which a function is defined.
- Key restrictions: no division by zero, no even roots of negatives, no logs of non-positive numbers.
- The natural domain is the maximal set; a restricted domain is a deliberate subset.
- Domain of $f \pm g$, $f \cdot g$: $\text{dom}(f) \cap \text{dom}(g)$.
- Domain of $f/g$: $\text{dom}(f) \cap \text{dom}(g) \setminus \{x : g(x) = 0\}$.
- Domain of $f \circ g$: $\{x \in \text{dom}(g) : g(x) \in \text{dom}(f)\}$.
- In ML, the training data domain defines where a model is reliable; OOD detection safeguards against extrapolation.
- Domain adaptation addresses distribution shifts between training and deployment domains.
