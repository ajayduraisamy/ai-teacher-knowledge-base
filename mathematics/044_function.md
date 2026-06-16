# Concept: Function

## Concept ID

MATH-044

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define a function as a mapping from one set to another where each input maps to exactly one output.
- Use function notation $f(x)$ correctly and distinguish between independent and dependent variables.
- Apply the vertical line test to determine whether a relation is a function.
- Identify the domain and codomain of a given function.
- Recognize how functions serve as the foundational building block of neural networks and AI/ML systems.

## Prerequisites

- Basic algebra: ability to evaluate algebraic expressions for given numeric values.
- Familiarity with sets and set notation (e.g., $\mathbb{R}$, interval notation).
- Understanding of ordered pairs $(x, y)$ and the Cartesian coordinate plane.
- Elementary graphing: plotting points and drawing curves on the $xy$-plane.

## Definition

A **function** is a rule that assigns to every element $x$ in a set $A$ (called the **domain**) exactly one element $y$ in a set $B$ (called the **codomain**). We write:

$$f: A \to B$$

and read it as "f maps A to B." For a particular input $x \in A$, the corresponding output is written as $f(x)$, read as "f of x."

The defining property of a function is **uniqueness of output**: for a given $x$, there cannot be two different values of $f(x)$. If there is any $x$ in $A$ that maps to more than one $y$, the rule is not a function — it is called a **relation** instead.

A function can be represented in multiple ways: as an equation ($f(x) = x^2 + 1$), as a table of input-output pairs, as a graph on the coordinate plane, as a verbal description, or as a set of ordered pairs.

## Intuition

Think of a function as a **machine** or a **processor**. You feed an input into the machine, the machine performs a fixed operation, and it produces exactly one output. If you feed the same input again, you always get the same output.

For example, consider a vending machine: you press a button (input) and the machine gives you exactly one snack (output). The same button always gives the same snack. If a machine sometimes gave you chips and sometimes gave you a chocolate bar for the same button press, it would not be a well-defined function.

Another intuition: a function is like a **mathematical contract** — for every input in the domain, the function promises to deliver exactly one output. It does not promise that all outputs are unique (different inputs can give the same output), and it does not promise that every element of the codomain is actually used. It only promises that each input gets a single, well-defined output.

## Why This Concept Matters

Functions are the single most important concept in all of mathematics. They describe how quantities depend on one another. In physics, position is a function of time. In economics, profit is a function of price and quantity. In biology, population size is a function of available resources.

In artificial intelligence and machine learning, **everything is a function**:

- A neural network is a function $f_\theta(x)$ parameterized by weights $\theta$ that maps input data $x$ to predictions $\hat{y}$.
- A loss function $L(\hat{y}, y)$ maps predictions and ground truth to a scalar error.
- An activation function like ReLU or sigmoid maps a neuron's pre-activation value to its post-activation value.
- The training process itself finds the parameters $\theta$ that minimize the loss — that is, it finds the function that best fits the data.

Without a deep understanding of functions, it is impossible to understand how models make predictions, how they are trained, or how they can be improved.

## Historical Background

The concept of a function evolved over centuries. Ancient mathematicians like Euclid and Archimedes worked with relationships between quantities (e.g., the area of a circle as a function of its radius) but did not formalize the notion.

In the 17th century, Gottfried Wilhelm Leibniz first used the word "function" (from Latin *functio*, meaning "performance" or "execution") to describe a quantity that depends on a variable. Leonhard Euler in the 18th century introduced the notation $f(x)$ that we use today. Euler defined a function as "an analytic expression composed in any way of a variable quantity and numbers or constant quantities."

The modern definition — a function as a set of ordered pairs with the uniqueness property — was developed in the 19th century by mathematicians including Peter Dirichlet, Bernhard Riemann, and Georg Cantor as part of the rigorization of analysis. Dirichlet's definition of a function as "a rule that assigns a unique value to each element of a domain" is essentially the definition we use today.

In the 20th century, the function concept became central to computer science through the theory of computation (Alonzo Church's lambda calculus, Alan Turing's computable functions) and later to machine learning through the concept of function approximation.

## Real World Examples

**Example 1: Currency Conversion.** The function $f(x) = 0.92x$ converts an amount $x$ in US dollars to euros (at a rate of 0.92 EUR per USD). Input: dollars. Output: euros. Each dollar amount maps to exactly one euro amount.

**Example 2: Temperature Conversion.** The function $C(F) = \frac{5}{9}(F - 32)$ converts Fahrenheit to Celsius. For $F = 212$, $C(212) = 100^\circ$C. For $F = 32$, $C(32) = 0^\circ$C.

**Example 3: Shipping Cost.** A shipping company charges $f(w) = 5 + 2w$ dollars to ship a package of weight $w$ kg (where $w \leq 20$). The base cost is $5, and each additional kilogram costs $2.

**Example 4: Population Growth.** The size of a bacterial colony after $t$ hours is $P(t) = 100 \cdot 2^t$ (starting with 100 bacteria that double every hour). At $t = 0$, $P(0) = 100$. At $t = 3$, $P(3) = 800$.

**Example 5: Exam Scoring.** A teacher's grading rubric defines a function $g(s)$ that maps a raw score $s$ (out of 100) to a letter grade. For example, $g(95) = A$, $g(85) = B$, $g(75) = C$, $g(65) = D$, $g(50) = F$. This is a piecewise-defined function.

## AI/ML Relevance

Functions are the language in which every machine learning model is expressed. Here are the most critical connections:

**1. Neural Networks as Universal Function Approximators.** The Universal Approximation Theorem states that a feedforward neural network with a single hidden layer containing enough neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to any desired degree of accuracy. This means neural networks are not just functions — they are **function approximators** that can learn any input-output relationship from data.

**2. Activation Functions.** Every neuron in a neural network applies an activation function to its weighted sum of inputs. Common activation functions include:

| Activation | Formula | Output Range |
|---|---|---|
| ReLU | $f(x) = \max(0, x)$ | $[0, \infty)$ |
| Sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ |
| Softmax | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$, sums to 1 |

**3. Loss Functions.** Training a model requires a loss function that measures prediction error:

- Mean Squared Error: $L(\hat{y}, y) = \frac{1}{n}\sum(\hat{y}_i - y_i)^2$ (for regression)
- Cross-Entropy Loss: $L(\hat{y}, y) = -\sum y_i \log(\hat{y}_i)$ (for classification)

**4. Gradient Descent.** The gradient $\nabla f(x)$ is a vector of partial derivatives that tells us the direction of steepest ascent. We train models by descending along the negative gradient — an inherently function-based process.

**5. Decision Functions.** In classification, a model learns a decision function $f: \mathbb{R}^n \to \{0, 1, \ldots, K-1\}$ that maps feature vectors to class labels. In regression, the function maps to real numbers.

**6. Kernel Functions.** In SVM, kernel functions like $K(x, x') = \exp(-\gamma\|x - x'\|^2)$ (RBF kernel) map data into a higher-dimensional feature space where it becomes linearly separable.

**7. Probability Density Functions.** Generative models like VAEs and GANs learn probability density functions $p(x)$ or $p(x|z)$ that describe the distribution of data.

## Mathematical Explanation

A function $f: A \to B$ consists of three components:

1. **Domain** $A$: the set of all allowable inputs.
2. **Codomain** $B$: the set that contains all possible outputs (but not every element of $B$ must be used).
3. **Rule** $f$: a mapping that assigns to each $x \in A$ exactly one $y \in B$.

The **range** (or **image**) of $f$ is the subset of $B$ that is actually attained: $\{f(x) : x \in A\}$. The range is a subset of the codomain.

**Function Notation:** If $f(x) = x^2$, then:
- $f(3) = 3^2 = 9$
- $f(-2) = (-2)^2 = 4$
- $f(a) = a^2$
- $f(a + h) = (a + h)^2$

**Independent and Dependent Variables:** In $y = f(x)$, $x$ is the **independent variable** (we choose its value) and $y$ is the **dependent variable** (its value depends on $x$).

**Types of Functions:**
- **Polynomial:** $f(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_0$
- **Rational:** $f(x) = \frac{p(x)}{q(x)}$ where $p, q$ are polynomials
- **Trigonometric:** $f(x) = \sin x$, $\cos x$, $\tan x$
- **Exponential:** $f(x) = a^x$ where $a > 0$
- **Logarithmic:** $f(x) = \log_a x$
- **Piecewise:** defined by different expressions on different intervals

**Vertical Line Test:** A curve in the $xy$-plane represents a function $y = f(x)$ if and only if any vertical line intersects the curve at most once. This tests the uniqueness condition: if a vertical line hits the curve twice, then one $x$-value has two $y$-values, violating the definition.

## Formula(s)

**General function notation:**
$$f: A \to B, \quad f(x) = y$$

**Linear function:**
$$f(x) = mx + b$$
Where $m$ is the slope and $b$ is the $y$-intercept.

**Quadratic function:**
$$f(x) = ax^2 + bx + c, \quad a \neq 0$$

**Exponential function:**
$$f(x) = a \cdot b^x, \quad b > 0, \, b \neq 1$$

**Logarithmic function:**
$$f(x) = \log_b(x), \quad b > 0, \, b \neq 1, \, x > 0$$

**Power function:**
$$f(x) = x^n$$

**Absolute value function:**
$$f(x) = |x| = \begin{cases} x, & x \geq 0 \\ -x, & x < 0 \end{cases}$$

## Properties

1. **Determinism:** A function always produces the same output for the same input. If $x_1 = x_2$, then $f(x_1) = f(x_2)$.

2. **Uniqueness:** Each input maps to exactly one output. This is the defining property.

3. **Domain-specific:** A function is only defined for inputs within its domain. Evaluating $f(x)$ for $x$ outside the domain is undefined.

4. **One-to-one (Injectivity):** A function is injective if different inputs always produce different outputs: $x_1 \neq x_2 \implies f(x_1) \neq f(x_2)$. Not all functions are injective; for example, $f(x) = x^2$ is not injective because $f(2) = f(-2) = 4$.

5. **Onto (Surjectivity):** A function is surjective if every element of the codomain is mapped to by at least one element of the domain: $\forall y \in B, \exists x \in A$ such that $f(x) = y$.

6. **Bijectivity:** A function that is both injective and surjective is called bijective. Bijective functions have inverses.

7. **Monotonicity:** A function is increasing if $x_1 < x_2 \implies f(x_1) < f(x_2)$ and decreasing if $x_1 < x_2 \implies f(x_1) > f(x_2)$.

8. **Even and Odd:** $f$ is even if $f(-x) = f(x)$ (symmetric about the $y$-axis). $f$ is odd if $f(-x) = -f(x)$ (symmetric about the origin).

9. **Periodicity:** $f$ is periodic if $f(x + T) = f(x)$ for some period $T > 0$. Trigonometric functions are periodic.

## Step-by-Step Worked Examples

### Example 1: Evaluating a Function at Given Points

Given $f(x) = 3x^2 - 2x + 1$, evaluate $f(0)$, $f(2)$, $f(-1)$, and $f(a + h)$.

**Step 1:** Evaluate $f(0)$ by substituting $x = 0$:
$$f(0) = 3(0)^2 - 2(0) + 1 = 0 - 0 + 1 = 1$$

**Step 2:** Evaluate $f(2)$ by substituting $x = 2$:
$$f(2) = 3(2)^2 - 2(2) + 1 = 3(4) - 4 + 1 = 12 - 4 + 1 = 9$$

**Step 3:** Evaluate $f(-1)$ by substituting $x = -1$:
$$f(-1) = 3(-1)^2 - 2(-1) + 1 = 3(1) + 2 + 1 = 3 + 2 + 1 = 6$$

**Step 4:** Evaluate $f(a + h)$ by substituting $x = a + h$:
$$f(a + h) = 3(a + h)^2 - 2(a + h) + 1 = 3(a^2 + 2ah + h^2) - 2a - 2h + 1$$
$$= 3a^2 + 6ah + 3h^2 - 2a - 2h + 1$$

**Answers:** $f(0) = 1$, $f(2) = 9$, $f(-1) = 6$, $f(a + h) = 3a^2 + 6ah + 3h^2 - 2a - 2h + 1$

### Example 2: Determining Whether a Relation is a Function

Determine whether each of the following relations represents a function from $A = \{1, 2, 3\}$ to $B = \{a, b, c, d\}$.

**Relation R1:** $\{(1, a), (2, b), (3, c)\}$

**Step 1:** Check that every element of $A$ appears as a first component. Yes: 1, 2, 3 all appear.
**Step 2:** Check that no element of $A$ maps to more than one element of $B$. Each of 1, 2, 3 appears exactly once.
**Conclusion:** R1 is a function.

**Relation R2:** $\{(1, a), (1, b), (2, c), (3, d)\}$

**Step 1:** Every element of $A$ appears. Yes.
**Step 2:** Element 1 appears twice: mapping to both $a$ and $b$. This violates the uniqueness condition.
**Conclusion:** R2 is NOT a function.

**Relation R3:** $\{(1, a), (2, b)\}$

**Step 1:** Element 3 of $A$ does not appear at all. The domain of the relation is $\{1, 2\} \neq A$.
**Conclusion:** R3 is NOT a function from $A$ to $B$ (though it is a function from $\{1, 2\}$ to $B$).

### Example 3: Vertical Line Test

Determine whether each graph represents a function $y = f(x)$.

**Graph A:** A parabola opening upward: $y = x^2$.

**Step 1:** Draw several vertical lines across the graph.
**Step 2:** Observe that each vertical line intersects the parabola at exactly one point.
**Conclusion:** The parabola $y = x^2$ passes the vertical line test. It IS a function.

**Graph B:** A circle: $x^2 + y^2 = 4$.

**Step 1:** Draw a vertical line at $x = 0$. It intersects the circle at $(0, 2)$ and $(0, -2)$ — two points!
**Step 2:** Draw a vertical line at $x = 1$. It intersects at $(1, \sqrt{3})$ and $(1, -\sqrt{3})$ — two points.
**Conclusion:** The circle fails the vertical line test. It is NOT a function (it is a relation).

**Graph C:** A horizontal line: $y = 3$.

**Step 1:** Draw a vertical line at any $x$. It intersects at exactly one point: $(x, 3)$.
**Conclusion:** The horizontal line passes the vertical line test. It IS a function (a constant function).

### Example 4: Evaluating a Piecewise Function

Given $f(x) = \begin{cases} x^2, & x < 0 \\ 2x + 1, & 0 \leq x < 3 \\ 5, & x \geq 3 \end{cases}$, evaluate $f(-2)$, $f(0)$, $f(2)$, and $f(5)$.

**Step 1:** For $f(-2)$, note $-2 < 0$, so use $f(x) = x^2$:
$$f(-2) = (-2)^2 = 4$$

**Step 2:** For $f(0)$, note $0 \leq 0 < 3$, so use $f(x) = 2x + 1$:
$$f(0) = 2(0) + 1 = 1$$

**Step 3:** For $f(2)$, note $0 \leq 2 < 3$, so use $f(x) = 2x + 1$:
$$f(2) = 2(2) + 1 = 5$$

**Step 4:** For $f(5)$, note $5 \geq 3$, so use $f(x) = 5$:
$$f(5) = 5$$

**Answers:** $f(-2) = 4$, $f(0) = 1$, $f(2) = 5$, $f(5) = 5$

### Example 5: Finding the Difference Quotient

For $f(x) = x^2 + 3x$, find and simplify the difference quotient $\frac{f(x + h) - f(x)}{h}$, $h \neq 0$.

**Step 1:** Compute $f(x + h)$:
$$f(x + h) = (x + h)^2 + 3(x + h) = x^2 + 2xh + h^2 + 3x + 3h$$

**Step 2:** Compute $f(x + h) - f(x)$:
$$f(x + h) - f(x) = (x^2 + 2xh + h^2 + 3x + 3h) - (x^2 + 3x)$$
$$= x^2 + 2xh + h^2 + 3x + 3h - x^2 - 3x$$
$$= 2xh + h^2 + 3h$$

**Step 3:** Divide by $h$:
$$\frac{f(x + h) - f(x)}{h} = \frac{2xh + h^2 + 3h}{h} = 2x + h + 3$$

**Step 4:** Take the limit as $h \to 0$ to find the derivative:
$$f'(x) = \lim_{h \to 0} (2x + h + 3) = 2x + 3$$

**Answer:** The difference quotient simplifies to $2x + h + 3$. The derivative is $f'(x) = 2x + 3$.

### Example 6: Function Representation (Table, Equation, Graph)

Represent the function $f(x) = 2x - 1$ as a table of values for $x = -2, -1, 0, 1, 2$ and describe its graph.

**Step 1:** Compute the table:

| $x$ | $f(x) = 2x - 1$ |
|---|---|
| $-2$ | $2(-2) - 1 = -5$ |
| $-1$ | $2(-1) - 1 = -3$ |
| $0$ | $2(0) - 1 = -1$ |
| $1$ | $2(1) - 1 = 1$ |
| $2$ | $2(2) - 1 = 3$ |

**Step 2:** The points are $(-2, -5)$, $(-1, -3)$, $(0, -1)$, $(1, 1)$, $(2, 3)$.

**Step 3:** The graph is a straight line with slope $m = 2$ and $y$-intercept $b = -1$. It passes the vertical line test. The domain is $\mathbb{R}$ and the range is $\mathbb{R}$.

## Visual Interpretation

A function can be visualized in several powerful ways:

**Mapping Diagram:** Draw two ovals (sets $A$ and $B$). For each element $x \in A$, draw an arrow from $x$ to $f(x) \in B$. A function has exactly one arrow leaving each element of $A$. Multiple arrows can point to the same element of $B$, and some elements of $B$ may have no arrows pointing to them.

**Graph on the Coordinate Plane:** Plot the set of points $\{(x, f(x)) : x \in A\}$. The vertical line test is the definitive visual check: if any vertical line touches the graph more than once, it is not a function.

**Function Machine:** Imagine a box labelled $f$ with an input chute and an output chute. A number $x$ goes in, the machine applies the rule, and $f(x)$ comes out. This emphasizes the input-output nature.

**Example Graph — Linear Function $f(x) = 2x + 1$:**

```
y
|
|   /
|  /  (2, 5)
| /   (1, 3)
|/    (0, 1)
|----(0, 1)--- x
|    (-1, -1)
|
```

The line has slope 2 (rise 2, run 1) and crosses the $y$-axis at $(0, 1)$.

**Example Graph — Quadratic Function $f(x) = x^2$:**

```
y
|     *
|   *   *
| *       *
|*         *
|----------- x
```

A parabola symmetric about the $y$-axis. The vertex is at $(0, 0)$. For any $x \neq 0$, two different $x$ values map to the same $y$, so $x^2$ is not injective, but it still passes the vertical line test.

## Common Mistakes

1. **Confusing the vertical line test with the horizontal line test.** The vertical line test checks whether a relation is a function (each $x$ maps to one $y$). The horizontal line test checks whether a function is one-to-one/injective. These are different concepts. A parabola $y = x^2$ passes the vertical line test (it is a function) but fails the horizontal line test (it is not one-to-one).

2. **Assuming all relations are functions.** Many students think that any equation relating $x$ and $y$ is automatically a function. The circle $x^2 + y^2 = 1$ is a counterexample: a single $x$ value (except $\pm 1$) corresponds to two $y$ values.

3. **Forgetting that $f(x)$ is not $f$ times $x$.** The notation $f(x)$ is function application, not multiplication. $f(x)$ means "the output of function $f$ when the input is $x$." It has nothing to do with multiplying $f$ by $x$.

4. **Evaluating a function at values outside its domain.** For $f(x) = \sqrt{x - 1}$, computing $f(0)$ gives $\sqrt{-1}$, which is undefined in the real numbers. Always check that the input lies in the domain before evaluating.

5. **Confusing independent and dependent variables.** In $y = f(x)$, $x$ is the independent variable (we choose it freely from the domain) and $y$ is the dependent variable (its value is determined by $x$). Swapping these roles leads to incorrect reasoning.

6. **Thinking that $f(x)$ always outputs a single number.** While a function always outputs a single value, that value can be a vector, matrix, or another complex object. For example, a neural network $f(x)$ may output a probability vector over 10 classes.

7. **Assuming $f(x+1) = f(x) + f(1)$.** This is only true for linear functions of the form $f(x) = ax$. For example, $f(x) = x^2$ gives $f(x+1) = (x+1)^2 = x^2 + 2x + 1 \neq x^2 + 1 = f(x) + f(1)$.

8. **Misinterpreting piecewise functions.** A piecewise function is still a single function — it just has different rules for different parts of the domain. Students sometimes treat it as multiple separate functions.

## Interview Questions

### Beginner

1. **What is a function? Give the formal definition.**
   *Answer: A function $f: A \to B$ is a rule that assigns to every element $x$ in the domain $A$ exactly one element $y$ in the codomain $B$. We write $y = f(x)$.*

2. **What does the vertical line test tell you?**
   *Answer: The vertical line test determines whether a graph represents a function. If any vertical line drawn on the graph intersects it at more than one point, the graph is not a function (it is a relation).*

3. **What is the difference between $f(x) = x^2$ and $f(x) = 2^x$?**
   *Answer: $f(x) = x^2$ is a quadratic (polynomial) function where the variable is the base. $f(x) = 2^x$ is an exponential function where the variable is the exponent. They grow very differently: quadratic grows polynomially, exponential grows much faster.*

4. **In the equation $y = 3x + 5$, which is the independent variable and which is the dependent variable?**
   *Answer: $x$ is the independent variable (we choose its value), and $y$ is the dependent variable (its value depends on $x$ through the function rule).*

5. **Can a function have two different inputs that give the same output? Provide an example.**
   *Answer: Yes. For $f(x) = x^2$, we have $f(2) = 4$ and $f(-2) = 4$. Two different inputs ($2$ and $-2$) give the same output ($4$). This is allowed for a general function.*

6. **What is the difference between a function and a relation?**
   *Answer: A relation is any set of ordered pairs. A function is a special type of relation where each input (first component) maps to exactly one output (second component). All functions are relations, but not all relations are functions.*

### Intermediate

1. **What is the difference between the codomain and the range of a function?**
   *Answer: The codomain $B$ is the set that the function is defined to map into; it is declared as part of the function definition. The range (or image) is the subset of the codomain that is actually attained: $\{f(x) : x \in A\}$. The range is always a subset of the codomain, and they are equal exactly when the function is surjective.*

2. **Explain why neural networks are called "universal function approximators." What are the conditions and limitations?**
   *Answer: The Universal Approximation Theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to any desired accuracy, provided the activation function is non-constant, bounded, and continuous (e.g., sigmoid). Limitations: the theorem does not guarantee that the network can be trained to find the right weights (it is an existence result, not a constructive one), it requires potentially exponentially many neurons, and it only applies to continuous functions on compact domains.*

3. **How does the activation function in a neural network relate to the concept of a mathematical function?**
   *Answer: An activation function is a mathematical function applied to the output of each neuron. Common activation functions include ReLU ($f(x) = \max(0, x)$), sigmoid ($\sigma(x) = 1/(1+e^{-x})$), and tanh. Each activation function has a specific domain ($\mathbb{R}$ for all of these) and range. They introduce non-linearity into the network, which is essential for learning complex patterns. Without non-linear activation functions, the entire network would collapse into a single linear transformation.*

4. **What is the difference between an injective, surjective, and bijective function? Why does bijectivity matter for invertibility?**
   *Answer: Injective (one-to-one): each output comes from at most one input. Surjective (onto): every element of the codomain is used. Bijective: both injective and surjective. A function has an inverse if and only if it is bijective. This is because the inverse needs to map each output back to exactly one input (requiring injectivity) and must be defined for every element of the codomain (requiring surjectivity).*

5. **What is the difference quotient $\frac{f(x+h) - f(x)}{h}$ and why is it important?**
   *Answer: The difference quotient computes the average rate of change of $f$ over an interval of length $h$. As $h \to 0$, it approaches the instantaneous rate of change, which is the derivative $f'(x)$. This is the foundation of differential calculus and is used in neural network training via backpropagation, which computes gradients by applying the chain rule to the difference quotient in the limit.*

6. **Given the function $f(x) = \frac{x^2 - 4}{x - 2}$, is this function defined at $x = 2$? Why or why not, and how can we "fix" it?**
   *Answer: $f(2)$ is undefined because the denominator becomes 0. However, for $x \neq 2$, $f(x) = \frac{(x-2)(x+2)}{x-2} = x + 2$. The function has a removable discontinuity at $x = 2$. We can define $g(x) = \begin{cases} \frac{x^2-4}{x-2}, & x \neq 2 \\ 4, & x = 2 \end{cases}$ to create a continuous function. This is called removing the singularity.*

### Advanced

1. **Define the Dirac delta "function." Is it truly a function in the standard sense? Explain.**
   *Answer: The Dirac delta $\delta(x)$ is not a function in the standard sense. It is defined by the property $\int_{-\infty}^{\infty} \delta(x) f(x) \, dx = f(0)$, which is not achievable by any ordinary function. It is a distribution (or generalized function) — a linear functional on a space of test functions. In machine learning, it appears in the context of probability density functions for continuous random variables, and in the theory of kernels and reproducing kernel Hilbert spaces. It fails the standard definition because it is not well-defined as a mapping from $\mathbb{R}$ to $\mathbb{R}$ (it is infinite at $x=0$ and zero elsewhere).*

2. **Explain the concept of a "function space" and give two examples relevant to machine learning.**
   *Answer: A function space is a set of functions that share certain properties, equipped with a vector space structure (functions can be added and multiplied by scalars). Examples: (1) $L^2(\mathbb{R}^n)$, the space of square-integrable functions, is the natural setting for kernel methods in machine learning — the reproducing kernel Hilbert space (RKHS) is a subspace of $L^2$ with additional structure. (2) The space of continuous functions $C([0,1]^n)$ is the setting of the Universal Approximation Theorem for neural networks. (3) Sobolev spaces, which consist of functions with integrable derivatives, are used in physics-informed neural networks (PINNs) and optimal transport.*

3. **Give a formal proof that the composition of two injective functions is injective.**
   *Proof: Let $f: B \to C$ and $g: A \to B$ be injective functions. We want to show $f \circ g: A \to C$ is injective. Suppose $(f \circ g)(x_1) = (f \circ g)(x_2)$ for some $x_1, x_2 \in A$. Then $f(g(x_1)) = f(g(x_2))$. Since $f$ is injective, $g(x_1) = g(x_2)$. Since $g$ is injective, $x_1 = x_2$. Therefore $f \circ g$ is injective.*

4. **A neural network $f_\theta: \mathbb{R}^n \to \mathbb{R}^m$ with ReLU activations is a piecewise linear function. Explain what this means and why it matters for optimization.**
   *Answer: A ReLU network partitions the input space $\mathbb{R}^n$ into convex polytopes (linear regions). Within each region, the network computes a distinct affine transformation $Wx + b$. The number of regions can grow exponentially with depth. This piecewise linear structure means: (1) the loss landscape is piecewise linear-quadratic, leading to many local minima, (2) gradients are piecewise constant (the ReLU derivative is 0 or 1), (3) the model can overfit if the number of regions is too large relative to the data, and (4) adversarial examples exist at the boundaries between regions.*

## Practice Problems

### Easy

1. Given $f(x) = 3x - 7$, evaluate $f(0)$, $f(2)$, and $f(-3)$.

2. Determine whether the set of ordered pairs $\{(1, 2), (2, 3), (3, 2), (4, 5)\}$ represents a function.

3. The graph of $x = y^2$ is a parabola opening to the right. Does this graph represent $y$ as a function of $x$? Apply the vertical line test.

4. For $f(x) = 2x^2 + 1$, find $f(-1)$, $f(0)$, and $f(3)$.

5. A function is defined by the table:

| $x$ | $f(x)$ |
|---|---|
| 0 | 5 |
| 1 | 8 |
| 2 | 11 |
| 3 | 14 |

What is the rule for this function?

### Medium

1. Find the domain of $f(x) = \sqrt{5 - 2x}$.

2. Given $f(x) = \begin{cases} x + 2, & x < 1 \\ 4x - 1, & x \geq 1 \end{cases}$, evaluate $f(-2)$, $f(1)$, and $f(3)$.

3. For $f(x) = x^2 - 4x + 3$, find and simplify $\frac{f(2 + h) - f(2)}{h}$, $h \neq 0$.

4. Determine whether $f(x) = \frac{x}{|x|}$ is a function. Evaluate $f(5)$, $f(-3)$, and $f(0)$. What is its domain?

5. The cost to produce $x$ widgets is $C(x) = 1000 + 5x$ dollars. The revenue from selling $x$ widgets is $R(x) = 12x$ dollars. Write the profit function $P(x) = R(x) - C(x)$ and find $P(200)$.

### Hard

1. Prove that the function $f(x) = \frac{2x + 3}{x - 2}$ is injective on its domain $(-\infty, 2) \cup (2, \infty)$.

2. A neural network with one hidden layer and sigmoid activation computes $f(x) = \sum_{i=1}^n w_i^{(2)} \sigma(w_i^{(1)} x + b_i^{(1)}) + b^{(2)}$, where $\sigma(z) = \frac{1}{1 + e^{-z}}$. Is $f$ a linear or non-linear function of $x$? Justify your answer. What happens to $f$ as $n \to \infty$?

3. Find all functions $f: \mathbb{R} \to \mathbb{R}$ such that $f(x + y) = f(x) + f(y)$ for all $x, y \in \mathbb{R}$ and $f(xy) = f(x)f(y)$ for all $x, y \in \mathbb{R}$. (These are called field automorphisms of $\mathbb{R}$.)

## Solutions

### Easy Solutions

**1.** $f(0) = 3(0) - 7 = -7$. $f(2) = 3(2) - 7 = 6 - 7 = -1$. $f(-3) = 3(-3) - 7 = -9 - 7 = -16$.

**2.** Each first component (1, 2, 3, 4) appears exactly once. No $x$-value maps to two different $y$-values. Therefore, this is a function.

**3.** The equation $x = y^2$ gives $x$ as a function of $y$, but for $y$ as a function of $x$ we apply the vertical line test. A vertical line at $x = 4$ intersects at $(4, 2)$ and $(4, -2)$ — two points. Therefore, $x = y^2$ does NOT represent $y$ as a function of $x$. (However, $y = \sqrt{x}$ and $y = -\sqrt{x}$ are two separate functions.)

**4.** $f(-1) = 2(-1)^2 + 1 = 2(1) + 1 = 3$. $f(0) = 2(0)^2 + 1 = 1$. $f(3) = 2(9) + 1 = 18 + 1 = 19$.

**5.** The outputs increase by 3 for each increase of 1 in $x$. The pattern is $f(x) = 3x + 5$. Check: $f(0) = 5$, $f(1) = 8$, $f(2) = 11$, $f(3) = 14$.

### Medium Solutions

**1.** The expression under the square root must be non-negative: $5 - 2x \geq 0$. So $-2x \geq -5$, which gives $x \leq \frac{5}{2}$. Domain: $(-\infty, \frac{5}{2}]$.

**2.** For $f(-2)$: $-2 < 1$, so use $f(x) = x + 2$: $f(-2) = -2 + 2 = 0$. For $f(1)$: $1 \geq 1$, so use $f(x) = 4x - 1$: $f(1) = 4(1) - 1 = 3$. For $f(3)$: $3 \geq 1$, so use $f(x) = 4x - 1$: $f(3) = 4(3) - 1 = 11$.

**3.** $f(2 + h) = (2 + h)^2 - 4(2 + h) + 3 = 4 + 4h + h^2 - 8 - 4h + 3 = h^2 - 1$. $f(2) = 4 - 8 + 3 = -1$. So $f(2+h) - f(2) = (h^2 - 1) - (-1) = h^2$. Therefore $\frac{f(2+h) - f(2)}{h} = \frac{h^2}{h} = h$ (for $h \neq 0$). The derivative at $x = 2$ is $\lim_{h \to 0} h = 0$.

**4.** $f(x) = \frac{x}{|x|}$ is the sign function (signum). $f(5) = \frac{5}{5} = 1$. $f(-3) = \frac{-3}{3} = -1$. $f(0)$ is undefined because division by zero. Domain: $(-\infty, 0) \cup (0, \infty)$. This is a function on its domain because each input maps to exactly one output.

**5.** $P(x) = R(x) - C(x) = 12x - (1000 + 5x) = 7x - 1000$. $P(200) = 7(200) - 1000 = 1400 - 1000 = 400$. The profit at 200 widgets is $400.

### Hard Solutions

**1.** To prove injectivity, we show $f(a) = f(b) \implies a = b$.

Assume $\frac{2a + 3}{a - 2} = \frac{2b + 3}{b - 2}$. Cross-multiply: $(2a + 3)(b - 2) = (2b + 3)(a - 2)$.
Expand LHS: $2ab - 4a + 3b - 6$.
Expand RHS: $2ab - 4b + 3a - 6$.
Subtract: $2ab - 4a + 3b - 6 = 2ab - 4b + 3a - 6$.
Cancel $2ab$ and $-6$: $-4a + 3b = -4b + 3a$.
Rearrange: $-4a - 3a = -4b - 3b$, so $-7a = -7b$, hence $a = b$.
Therefore $f$ is injective on its domain.

**2.** $f$ is a non-linear function of $x$ because the sigmoid activation $\sigma$ is non-linear. Even though the outer sum is linear, the inner terms $\sigma(w_i^{(1)} x + b_i^{(1)})$ are non-linear compositions. If $\sigma$ were linear (which it is not), the whole network would collapse to a linear function. As $n \to \infty$, by the Universal Approximation Theorem, $f$ can approximate any continuous function on a compact set arbitrarily well, given appropriate weights.

**3.** The conditions are Cauchy's functional equation (additivity) and multiplicativity. The only functions $f: \mathbb{R} \to \mathbb{R}$ satisfying both are $f(x) = 0$ for all $x$ (the zero function) and $f(x) = x$ for all $x$ (the identity function). 
Proof sketch: Additivity implies $f(q) = q f(1)$ for all rational $q$. If $f(1) = 0$, then $f(q) = 0$ for rationals, and multiplicativity forces $f(x) = 0$ for all reals. If $f(1) = 1$, then $f(q) = q$ for rationals. For any real $x$, using monotonicity derived from $f(x^2) = f(x)^2 \geq 0$ (so $f$ preserves order), we can show $f(x) = x$ for all reals. (Assuming the axiom of choice, there exist pathological non-measurable solutions, but they are highly non-constructive.)

## Related Concepts

- **Relation** — A function is a special type of relation. All functions are relations, but not all relations are functions.
- **Domain** — The set of all allowable inputs to a function.
- **Range** — The set of all outputs actually produced by a function.
- **Composite Function** — Combining two functions by applying one after the other: $(f \circ g)(x) = f(g(x))$.
- **Inverse Function** — A function that "undoes" another function: $f^{-1}(f(x)) = x$.
- **Linear Transformation** — A special type of function from vector spaces that preserves addition and scalar multiplication.
- **Derivative** — The derivative $f'(x)$ measures the instantaneous rate of change of $f$ at $x$.

## Next Concepts

- **Domain** (MATH-045) — A deeper look at finding domains of functions, including restrictions from square roots, denominators, and logarithms.
- **Range** (MATH-046) — Techniques for finding the set of all possible outputs of a function.
- **Composite Function** (MATH-047) — Building complex functions by chaining simpler ones.
- **Inverse Function** (MATH-048) — Finding functions that reverse the mapping of a given function.

## Summary

A function $f: A \to B$ is a rule that assigns to every element $x$ in its domain $A$ exactly one element $f(x)$ in its codomain $B$. Functions are the fundamental building blocks of mathematics, describing how one quantity depends on another. They can be represented as equations, tables, graphs, or verbal descriptions. The vertical line test determines whether a graph represents a function. Functions have properties like injectivity (one-to-one), surjectivity (onto), monotonicity, and parity. In AI and machine learning, neural networks are functions (universal function approximators), activation functions introduce non-linearity, loss functions measure error, and the training process optimizes a function to fit data.

## Key Takeaways

- A function maps each input to exactly one output — this is the defining property.
- The domain is the set of valid inputs; the codomain contains all possible outputs; the range is the subset of outputs actually used.
- The vertical line test checks whether a graph represents a function.
- Functions can be represented algebraically, numerically (tables), graphically, or verbally.
- A neural network is a function $f_\theta(x)$ parameterized by weights $\theta$ that maps inputs to predictions.
- Activation functions (ReLU, sigmoid, tanh) are mathematical functions that enable non-linear learning.
- The Universal Approximation Theorem guarantees that neural networks can approximate any continuous function.
- Understanding functions is prerequisite to understanding domain, range, composition, inverses, and calculus — all essential for machine learning.
