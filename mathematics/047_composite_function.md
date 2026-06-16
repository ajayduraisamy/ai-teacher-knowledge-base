# Concept: Composite Function

## Concept ID

MATH-047

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define the composite function $(f \circ g)(x) = f(g(x))$ and understand when it is defined.
- Determine the domain of a composite function.
- Decompose a complex function into a composition of simpler functions.
- Apply the chain rule for differentiating composite functions.
- Explain how deep neural networks are hierarchical compositions of simpler functions.
- Connect function composition to ML pipelines and end-to-end learned systems.

## Prerequisites

- Solid understanding of functions (MATH-044) including domain, range, and function notation.
- Understanding of domain (MATH-045) — composing functions creates new domain restrictions.
- Understanding of range (MATH-046) — the range of $g$ must intersect the domain of $f$ for $f \circ g$ to be defined.
- Basic algebra: substituting expressions into functions, simplifying algebraic expressions.
- Introduction to derivatives (for the chain rule section), but this is covered within the example.

## Definition

Given two functions $f: B \to C$ and $g: A \to B$, their **composition** is a new function $f \circ g: A \to C$ (read as $f$ composed with $g$) defined by:

$$(f \circ g)(x) = f(g(x))$$

The composition is defined only when the **range** of $g$ (the inner function) intersects the **domain** of $f$ (the outer function). In other words, $x$ must be in the domain of $g$, and $g(x)$ must be in the domain of $f$.

Composition is **not commutative**: $f \circ g$ is generally not equal to $g \circ f$. The order in which functions are composed matters critically.

Composition is **associative**: $(f \circ g) \circ h = f \circ (g \circ h)$. This allows us to compose long chains of functions without ambiguity.

## Intuition

Think of function composition as a **two-stage machine** or an **assembly line**. The output of the first machine becomes the input to the second machine.

Imagine a factory process:
1. Machine G takes raw material $x$ and produces a part $g(x)$.
2. Machine F takes the part $g(x)$ and finishes it into a final product $f(g(x))$.

The combined process is the composition $f \circ g$. You feed raw material $x$ at one end and get a finished product at the other.

Another analogy: **nested functions in a programming language**. In Python, `f(g(x))` first calls `g(x)`, takes its return value, and passes it to `f`. This is function composition.

A real-world example: converting Celsius to Fahrenheit and then interpreting that temperature:
$$C(F) = \frac{5}{9}(F - 32), \quad I(T) = \text{interpretation}(T)$$
$$(I \circ C)(F) = I(C(F)) = I\left(\frac{5}{9}(F - 32)\right)$$

## Why This Concept Matters

Function composition is the mathematical foundation of **building complex systems from simple components**:

**1. Neural Networks.** A deep neural network is literally a composition of functions:
$$f_{\text{NN}}(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$
where each layer $f_i$ applies an affine transformation followed by a non-linear activation. The depth of the network corresponds to the number of composed functions. Each layer transforms the representation, and the composition extracts increasingly abstract features.

**2. Chain Rule.** The derivative of a composition is given by the chain rule:
$$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$$
This is the foundation of **backpropagation** — the algorithm that trains neural networks by computing gradients through the composition of layers.

**3. ML Pipelines.** Machine learning systems often compose multiple functions:
$$p(x) = \text{softmax}(\text{logistic\_regression}(\text{PCA}(\text{standardize}(x))))$$
Each step transforms the data, and the pipeline as a whole is a composition of functions.

**4. Functional Programming.** The concept of composition is central to functional programming paradigms. Libraries like `scikit-learn`'s `Pipeline` class and Keras' `Sequential` model are built on the idea of function composition.

**5. Encoding-Decoding.** Autoencoders compose an encoder $E(x)$ and a decoder $D(z)$: $\hat{x} = D(E(x))$. The composition $D \circ E$ approximates the identity function.

## Historical Background

The notation $(f \circ g)(x)$ for function composition was introduced by the French mathematician Nicolas Bourbaki (a pseudonym for a group of 20th-century mathematicians) in the 1930s, though the concept has been used implicitly for centuries.

Gottfried Wilhelm Leibniz studied compositions of functions in the context of derivatives and discovered the chain rule in the 1670s. The chain rule in Leibniz notation, $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$, reflects the composability of derivatives.

The concept of functional composition became fundamental in the 20th century with the development of category theory, where categories are defined by objects and morphisms (functions) that compose associatively. Category theory is now influential in programming language theory and has applications in machine learning (e.g., categorical perspectives on generative models).

In computer science, the lambda calculus (developed by Alonzo Church in the 1930s) is built entirely on function abstraction and application. Composition is one of the fundamental operations.

## Real World Examples

**Example 1: Currency Conversion with Tax.** Suppose $g(x) = 0.92x$ converts USD to EUR (exchange rate), and $f(x) = 0.95x$ applies a 5% transaction fee. Then $(f \circ g)(x) = 0.95 \cdot 0.92x = 0.874x$ converts USD to EUR and deducts the fee. The order matters: $(g \circ f)(x) = 0.92 \cdot 0.95x = 0.874x$ gives the same result here because multiplication commutes, but in general order matters.

**Example 2: Temperature Conversion to Decision.** $C(F) = \frac{5}{9}(F - 32)$ converts Fahrenheit to Celsius. $D(C) = \begin{cases} \text{Cold}, & C \leq 0 \\ \text{Mild}, & 0 < C \leq 20 \\ \text{Hot}, & C > 20 \end{cases}$ decides comfort level. Then $(D \circ C)(F)$ tells us the comfort level directly from Fahrenheit.

**Example 3: Manufacturing Cost.** Suppose producing $x$ units requires $g(x) = 2x + 50$ hours of labor, and the cost per hour is $f(h) = 25h$ dollars. The total cost is $(f \circ g)(x) = 25(2x + 50) = 50x + 1250$ dollars.

**Example 4: Social Media.** A post $p$ gets $g(p)$ likes after 1 hour, and then the number of shares is $f(l) = 0.1l$ (10% of likes become shares). Then $(f \circ g)(p) = 0.1 \times g(p)$ gives the expected shares.

**Example 5: Medical Testing.** A diagnostic test has a sensitivity function $g(t)$ that gives the probability of detecting a disease given the true value $t$ of a biomarker. A treatment decision function $f(p)$ decides whether to treat based on the probability $p$. Then $(f \circ g)(t)$ gives the treatment decision based directly on the biomarker value.

## AI/ML Relevance

Function composition is arguably the most important mathematical concept in deep learning:

**1. Deep Neural Networks as Compositions.** A neural network with $L$ layers computes:
$$f(x) = f_L(f_{L-1}(\cdots f_1(x)\cdots))$$
where each $f_i$ consists of an affine transformation $W_i x + b_i$ followed by a non-linear activation $\sigma_i$:
$$f_i(z) = \sigma_i(W_i z + b_i)$$

The depth $L$ is the number of composed functions. This compositional structure is what gives deep networks their representational power — each layer builds on the representations learned by previous layers.

**2. Residual Networks (ResNets).** ResNets use a compositional structure with skip connections:
$$f_i(z) = z + \sigma_i(W_i z + b_i)$$
This is still a composition of functions but with an additive identity component that helps gradient flow.

**3. Chain Rule and Backpropagation.** Training neural networks requires computing gradients of the loss $L$ with respect to all parameters. Since the network is a composition:
$$L(\theta) = L(f_L(f_{L-1}(\cdots f_1(x)\cdots)))$$
the chain rule gives:
$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_{i+1}}{\partial f_i} \cdot \frac{\partial f_i}{\partial W_i}$$
This product of Jacobians is computed efficiently via backpropagation (reverse-mode automatic differentiation).

**4. ML Pipelines.** The scikit-learn `Pipeline` class composes data transformations with a final estimator:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])
pipeline.fit(X, y)
```
This is function composition in software engineering.

**5. Normalizing Flows.** A normalizing flow is a composition of invertible functions $f = f_K \circ \cdots \circ f_1$ that transforms a simple base distribution into a complex target distribution:
$$p(x) = p_0(f^{-1}(x)) \cdot \left|\det J_{f^{-1}}(x)\right|$$
The composition of invertible functions remains invertible, and the log-determinant of the Jacobian adds across the composition.

**6. Encoder-Decoder Architectures.** Autoencoders, VAEs, and many generative models compose an encoder $E: X \to Z$ and a decoder $D: Z \to X$:
$$\hat{x} = D(E(x))$$
The composition $D \circ E$ should approximate the identity (for autoencoders) or learn a meaningful latent representation.

**7. Attention and Transformers.** The Transformer architecture composes multiple functions: self-attention, feed-forward networks, layer normalization, and residual connections. Each block is itself a composition:
$$\text{block}(x) = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x + \text{SelfAttention}(x))))$$

## Mathematical Explanation

**Definition in Detail:**

Given $f: B \to C$ and $g: A \to B$, the composite $f \circ g: A \to C$ is defined by:
$$(f \circ g)(x) = f(g(x))$$

**Order Convention:** In $(f \circ g)(x)$, $g$ is applied first, then $f$. The composition is read from right to left: first $g$, then $f$.

**Domain of Composition:**
$$\text{dom}(f \circ g) = \{x \in \text{dom}(g) : g(x) \in \text{dom}(f)\}$$

For $(f \circ g)(x)$ to be defined:
1. $x$ must be in the domain of $g$.
2. $g(x)$ must be in the domain of $f$.

**Non-Commutativity:** In general, $f \circ g \neq g \circ f$. Examples:
- $f(x) = x^2$, $g(x) = x + 1$: $(f \circ g)(x) = (x+1)^2 = x^2 + 2x + 1$, $(g \circ f)(x) = x^2 + 1$. Not equal.
- When $f \circ g = g \circ f$, we say $f$ and $g$ commute.

**Associativity:** $(f \circ g) \circ h = f \circ (g \circ h)$. This allows us to write $f \circ g \circ h$ without parentheses.

**Multiple Compositions:** Composition can be extended to any number of functions:
$$(f_n \circ f_{n-1} \circ \cdots \circ f_1)(x) = f_n(f_{n-1}(\cdots f_1(x)\cdots))$$

**Decomposition:** Given a complex function $h(x)$, we can often decompose it as $h = f \circ g$ for simpler $f$ and $g$. For example:
- $h(x) = \sin(x^2)$: let $g(x) = x^2$ and $f(u) = \sin(u)$. Then $h = f \circ g$.
- $h(x) = e^{3x+1}$: let $g(x) = 3x+1$ and $f(u) = e^u$. Then $h = f \circ g$.

**Chain Rule (calculus):** If $h(x) = (f \circ g)(x) = f(g(x))$, then:
$$h'(x) = f'(g(x)) \cdot g'(x)$$

In Leibniz notation: if $y = f(u)$ and $u = g(x)$, then $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.

## Formula(s)

**Definition:**
$$(f \circ g)(x) = f(g(x))$$

**Domain:**
$$\text{dom}(f \circ g) = \{x \in \text{dom}(g) : g(x) \in \text{dom}(f)\}$$

**Chain Rule (single variable):**
$$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$$

**Chain Rule (multivariable):**
$$J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)$$
Where $J$ denotes the Jacobian matrix.

**Generalized chain rule for neural networks (backpropagation):**
For $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$:
$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial f_L} \cdot \prod_{k=L}^{i+1} \frac{\partial f_k}{\partial f_{k-1}} \cdot \frac{\partial f_i}{\partial W_i}$$

**Composition of linear functions:**
If $f(x) = Ax$ and $g(x) = Bx$, then $(f \circ g)(x) = A(Bx) = (AB)x$.

## Properties

1. **Non-commutative:** $f \circ g \neq g \circ f$ in general.

2. **Associative:** $(f \circ g) \circ h = f \circ (g \circ h)$.

3. **Identity function:** The identity function $id(x) = x$ satisfies $f \circ id = f = id \circ f$.

4. **Composition preserves injectivity:** If $f$ and $g$ are injective, then $f \circ g$ is injective.

5. **Composition preserves surjectivity:** If $f$ and $g$ are surjective, then $f \circ g$ is surjective.

6. **Composition of bijections is bijective:** The composition of two bijective functions is bijective.

7. **Inverse of composition:** $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$ (the reverse order).

8. **Domain restriction:** The domain of $f \circ g$ is always a subset of the domain of $g$.

9. **Range of composition:** $\text{range}(f \circ g) \subseteq \text{range}(f)$. The composition can only produce outputs that $f$ can produce from the values that $g$ attains.

## Step-by-Step Worked Examples

### Example 1: Basic Composition

Given $f(x) = x^2 + 1$ and $g(x) = 2x - 3$, find $(f \circ g)(x)$ and $(g \circ f)(x)$.

**Step 1:** Compute $(f \circ g)(x) = f(g(x))$.
$$f(g(x)) = f(2x - 3) = (2x - 3)^2 + 1$$

**Step 2:** Expand:
$$(2x - 3)^2 + 1 = (4x^2 - 12x + 9) + 1 = 4x^2 - 12x + 10$$

**Step 3:** Compute $(g \circ f)(x) = g(f(x))$.
$$g(f(x)) = g(x^2 + 1) = 2(x^2 + 1) - 3 = 2x^2 + 2 - 3 = 2x^2 - 1$$

**Answers:** $(f \circ g)(x) = 4x^2 - 12x + 10$, $(g \circ f)(x) = 2x^2 - 1$. They are not equal, demonstrating non-commutativity.

### Example 2: Domain of a Composite Function

Given $f(x) = \frac{1}{x}$ and $g(x) = \sqrt{x - 1}$, find $(f \circ g)(x)$ and its domain.

**Step 1:** Compute $(f \circ g)(x) = f(g(x)) = \frac{1}{\sqrt{x - 1}}$.

**Step 2:** Domain analysis:
- Domain of $g$: $\sqrt{x - 1}$ requires $x - 1 \geq 0 \implies x \geq 1$.
- For $f \circ g$, we also need $g(x) \neq 0$. $g(x) = \sqrt{x - 1} = 0$ when $x = 1$. So $x \neq 1$.

**Step 3:** Combine: $x > 1$.

**Answer:** $(f \circ g)(x) = \frac{1}{\sqrt{x - 1}}$, domain: $(1, \infty)$.

### Example 3: Decomposing a Function

Decompose $h(x) = e^{\cos(x^2)}$ into a composition of three functions $f$, $g$, and $k$ such that $h = f \circ g \circ k$.

**Step 1:** Look at the outermost operation. The outermost is exponentiation: $e^{\text{something}}$.
Let $f(u) = e^u$.

**Step 2:** The next inner operation is $\cos(\text{something})$.
Let $g(v) = \cos(v)$.

**Step 3:** The innermost operation is $x^2$.
Let $k(x) = x^2$.

**Step 4:** Verify:
$$(f \circ g \circ k)(x) = f(g(k(x))) = f(g(x^2)) = f(\cos(x^2)) = e^{\cos(x^2)} = h(x)$$

**Answer:** $k(x) = x^2$, $g(v) = \cos(v)$, $f(u) = e^u$ is one valid decomposition.

### Example 4: Chain Rule Application

Given $h(x) = \sin(3x^2 + 2)$, find $h'(x)$.

**Step 1:** Decompose $h$ as $f \circ g$ where $g(x) = 3x^2 + 2$ and $f(u) = \sin(u)$.

**Step 2:** Compute derivatives: $g'(x) = 6x$, $f'(u) = \cos(u)$.

**Step 3:** Apply chain rule: $h'(x) = f'(g(x)) \cdot g'(x) = \cos(3x^2 + 2) \cdot 6x$.

**Answer:** $h'(x) = 6x \cos(3x^2 + 2)$.

### Example 5: Composition with Tables

Functions $f$ and $g$ are defined by the tables below. Find $(f \circ g)(2)$ and $(g \circ f)(0)$.

$g$:
| $x$ | $g(x)$ |
|---|---|
| 0 | 2 |
| 1 | 3 |
| 2 | 4 |
| 3 | 1 |

$f$:
| $x$ | $f(x)$ |
|---|---|
| 1 | 5 |
| 2 | 7 |
| 3 | 9 |
| 4 | 11 |

**Step 1:** For $(f \circ g)(2)$, first find $g(2) = 4$. Then compute $f(4) = 11$.

**Step 2:** For $(g \circ f)(0)$, first find $f(0)$. But 0 is not in the domain of $f$! So $(g \circ f)(0)$ is undefined.

**Answers:** $(f \circ g)(2) = 11$. $(g \circ f)(0)$ is undefined.

### Example 6: Neural Network Layer Composition

A simple neural network has input $x \in \mathbb{R}^2$, hidden layer with weights $W_1 = \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix}$, bias $b_1 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$, ReLU activation, and output layer with weights $W_2 = \begin{pmatrix} 2 & -1 \end{pmatrix}$, bias $b_2 = 0$. Compute the forward pass for $x = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$.

**Step 1:** First layer (affine): $z_1 = W_1 x + b_1$.
$$z_1 = \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} \begin{pmatrix} 3 \\ 1 \end{pmatrix} + \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 3 + 2 \\ -3 + 0 \end{pmatrix} + \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 5 \\ -3 \end{pmatrix} + \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 6 \\ -4 \end{pmatrix}$$

**Step 2:** ReLU activation: $a_1 = \max(0, z_1)$.
$$a_1 = \begin{pmatrix} \max(0, 6) \\ \max(0, -4) \end{pmatrix} = \begin{pmatrix} 6 \\ 0 \end{pmatrix}$$

**Step 3:** Second layer (output): $z_2 = W_2 a_1 + b_2$.
$$z_2 = \begin{pmatrix} 2 & -1 \end{pmatrix} \begin{pmatrix} 6 \\ 0 \end{pmatrix} + 0 = 12 + 0 = 12$$

**Step 4:** The forward pass is the composition: $f(x) = f_2(f_1(x))$ where $f_1(x) = \max(0, W_1 x + b_1)$ and $f_2(u) = W_2 u + b_2$.

**Answer:** $f(x) = 12$.

### Example 7: Verifying Inverse via Composition

Verify that $f(x) = 2x + 3$ and $g(x) = \frac{x - 3}{2}$ are inverses by checking $f(g(x)) = x$ and $g(f(x)) = x$.

**Step 1:** Compute $f(g(x))$:
$$f(g(x)) = f\left(\frac{x - 3}{2}\right) = 2\left(\frac{x - 3}{2}\right) + 3 = (x - 3) + 3 = x$$

**Step 2:** Compute $g(f(x))$:
$$g(f(x)) = g(2x + 3) = \frac{(2x + 3) - 3}{2} = \frac{2x}{2} = x$$

**Step 3:** Since both compositions give the identity function, $f$ and $g$ are inverses.

**Answer:** Verified: $f \circ g = g \circ f = id$.

## Visual Interpretation

Function composition can be visualized in several ways:

**Mapping Diagram:**
```
x  $\to$  g  $\to$  g(x)  $\to$  f  $\to$  f(g(x))
         (intermediate)
```

The input flows from left to right, passing through each function in sequence. This is sometimes called a flow diagram or data flow diagram.

**Chain of Transformations:**

For $h = f \circ g \circ k$:

```
  k       g       f
x $\to$ k(x) $\to$ g(k(x)) $\to$ f(g(k(x)))
```

Each function transforms the data, and the output of one becomes the input to the next.

**Graphical View — Vertical Stacking:**

To visualize $(f \circ g)(x)$ graphically:
1. On the $x$-axis, locate $x$.
2. Move up to the graph of $g$ to find $g(x)$ on the $y$-axis.
3. From $g(x)$ on the $y$-axis, move horizontally to the graph of $f$.
4. Move vertically to the $y$-axis to read $f(g(x))$.

**Neural Network Architecture Diagram:**

```
Input $\to$ [Layer 1] $\to$ [Activation] $\to$ [Layer 2] $\to$ [Activation] $\to$ ... $\to$ [Output]
  x      f_1(x)       $\sigma$(f_1(x))     f_2($\sigma$(f_1(x)))     ...     f_L(...)
```

Each box represents a function, and the data flows left to right through the composition.

**Tree Diagram for Decomposition:**

For $h(x) = e^{\cos(x^2)}$:
```
      h
     / \
    e  cos
        / \
       ^2  x
```

The tree shows the hierarchical structure: $x$ is squared, then cosine is applied, then exponentiation.

## Common Mistakes

1. **Getting the order wrong.** $(f \circ g)(x)$ means apply $g$ first, then $f$. Students often mistakenly compute $f(x)$ first. Remember: $f \circ g$ is read as $f$ after $g$ — $g$ happens first.

2. **Assuming composition commutes.** $f \circ g$ is almost never equal to $g \circ f$. For example, $f(x) = x^2$ and $g(x) = x + 1$ give $f(g(x)) = (x+1)^2$ and $g(f(x)) = x^2 + 1$, which are different.

3. **Forgetting domain restrictions in composition.** When composing $f(g(x))$, you must ensure: (a) $x$ is in the domain of $g$, and (b) $g(x)$ is in the domain of $f$. For $f(x) = \sqrt{x}$ and $g(x) = x - 3$, the domain of $f \circ g$ is $x \geq 3$, not all real numbers.

4. **Misapplying the chain rule.** The correct form is $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$. A common mistake is $f'(g(x))$ alone (forgetting to multiply by $g'(x)$). Another is $f'(x) \cdot g'(x)$ (evaluating $f'$ at $x$ instead of $g(x)$).

5. **Confusing notation.** $f \circ g$ is a single function. $(f \circ g)(x)$ is its value at $x$. The circle is not multiplication — it is composition.

6. **Overlooking that $f(g(x))$ may be defined even when $g \circ f$ is not.** For $f(x) = \sqrt{x}$ and $g(x) = x^2$, $f(g(x)) = \sqrt{x^2} = |x|$ has domain $\mathbb{R}$. But $g(f(x)) = (\sqrt{x})^2 = x$ has domain $[0, \infty)$.

7. **Treating the inverse of a composition incorrectly.** $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$, not $f^{-1} \circ g^{-1}$. The order reverses.

8. **Thinking any two functions can be composed.** For $f \circ g$ to be defined, the range of $g$ must be a subset of (or at least intersect) the domain of $f$. If $g(x) \notin \text{dom}(f)$ for all $x$ in $\text{dom}(g)$, the composition is empty.

## Interview Questions

### Beginner

1. **What is a composite function? Give an example.**
   *Answer: A composite function $(f \circ g)(x) = f(g(x))$ applies $g$ first then $f$. Example: $f(x) = x^2$, $g(x) = x + 1$, then $(f \circ g)(x) = (x+1)^2$.*

2. **Given $f(x) = 2x$ and $g(x) = x + 3$, find $f(g(4))$ and $g(f(4))$.**
   *Answer: $f(g(4)) = f(7) = 14$. $g(f(4)) = g(8) = 11$. They are different, showing composition is not commutative.*

3. **What is the domain of $(f \circ g)(x)$ if $f(x) = \frac{1}{x}$ and $g(x) = x + 2$?**
   *Answer: $g(x) = x + 2$ has domain $\mathbb{R}$. $f$ requires $g(x) \neq 0$, i.e., $x + 2 \neq 0 \implies x \neq -2$. Domain: $(-\infty, -2) \cup (-2, \infty)$.*

4. **Explain why $(f \circ g)(x)$ is not the same as $(g \circ f)(x)$ in general.**
   *Answer: $(f \circ g)(x)$ means first apply $g$, then apply $f$ to the result. $(g \circ f)(x)$ means first apply $f$, then apply $g$. Since $f$ and $g$ are different functions, the order typically produces different results.*

5. **If $f(x) = \sqrt{x}$ and $g(x) = x^2$, evaluate $(f \circ g)(-3)$ and $(g \circ f)(-3)$ if possible.**
   *Answer: $(f \circ g)(-3) = f(g(-3)) = f(9) = \sqrt{9} = 3$. $(g \circ f)(-3) = g(f(-3))$ is undefined because $f(-3) = \sqrt{-3}$ is not a real number.*

6. **What does it mean for functions $f$ and $g$ to be inverses in terms of composition?**
   *Answer: $f$ and $g$ are inverses if $f(g(x)) = x$ for all $x$ in the domain of $g$, and $g(f(x)) = x$ for all $x$ in the domain of $f$. In composition notation: $f \circ g = id$ and $g \circ f = id$.*

### Intermediate

1. **Given $f(x) = x^2 + 1$ and $g(x) = \sqrt{x - 1}$, find $f(g(x))$ and $g(f(x))$. Simplify and state the domains.**
   *Answer: $f(g(x)) = (\sqrt{x-1})^2 + 1 = x - 1 + 1 = x$, domain: $[1, \infty)$. $g(f(x)) = \sqrt{(x^2 + 1) - 1} = \sqrt{x^2} = |x|$, domain: $\mathbb{R}$. Interesting: $f \circ g = id$ on $[1, \infty)$, but $g \circ f \neq id$ (it gives $|x|$, not $x$).*

2. **Explain how backpropagation uses the chain rule to compute gradients in a neural network.**
   *Answer: A neural network computes a composition $f = f_L \circ \cdots \circ f_1$. The loss $L(f(x))$ is also a composition. Backpropagation applies the chain rule from the output backward: $\frac{\partial L}{\partial f_i} = \frac{\partial L}{\partial f_{i+1}} \cdot \frac{\partial f_{i+1}}{\partial f_i}$. Each layer gradient uses the gradient from the layer above, multiplied by the local Jacobian. This is efficient because it reuses computations.*

3. **Decompose $h(x) = \frac{1}{1 + e^{-(ax + b)}}$ into a composition of elementary functions. What ML function does this represent?**
   *Answer: $g_1(x) = ax + b$ (affine), $g_2(u) = -u$ (negation), $g_3(v) = e^v$ (exponential), $g_4(w) = 1 + w$ (add 1), $g_5(z) = 1/z$ (reciprocal). Then $h = g_5 \circ g_4 \circ g_3 \circ g_2 \circ g_1$. This is the sigmoid function $\sigma(ax + b)$, a single neuron with sigmoid activation.*

4. **Given $f(x) = 2x - 1$ and $g(x) = \frac{x+1}{2}$, verify that $f \circ g = g \circ f = id$. What does this tell us?**
   *Answer: $f(g(x)) = 2(\frac{x+1}{2}) - 1 = x + 1 - 1 = x$. $g(f(x)) = \frac{(2x-1)+1}{2} = \frac{2x}{2} = x$. Since both compositions equal the identity, $f$ and $g$ are inverses of each other.*

5. **What is the domain of $(f \circ g \circ h)(x)$ where $f(x) = \frac{1}{x}$, $g(x) = \sqrt{x}$, and $h(x) = x - 2$?**
   *Answer: $h$ has domain $\mathbb{R}$, $g$ requires $h(x) \geq 0 \implies x \geq 2$, $f$ requires $g(h(x)) \neq 0 \implies \sqrt{x-2} \neq 0 \implies x \neq 2$. Combining: $x > 2$. Domain: $(2, \infty)$.*

6. **Explain what a residual connection $f(x) = x + g(x)$ accomplishes from the perspective of function composition.**
   *Answer: A residual connection adds an identity skip connection alongside $g$. Instead of learning $g(x)$ directly, the layer learns $f(x) = x + g(x)$. During backpropagation, the chain rule gives $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot (1 + \frac{\partial g}{\partial x})$. The additive $1$ ensures the gradient never completely vanishes, mitigating the vanishing gradient problem.*

### Advanced

1. **Prove that the composition of two injective functions is injective.**
   *Proof: Let $f: B \to C$ and $g: A \to B$ be injective. Take $x_1, x_2 \in A$ with $(f \circ g)(x_1) = (f \circ g)(x_2)$. Then $f(g(x_1)) = f(g(x_2))$. Since $f$ is injective, $g(x_1) = g(x_2)$. Since $g$ is injective, $x_1 = x_2$. Therefore $f \circ g$ is injective.*

2. **A deep neural network $f(x) = f_L \circ \cdots \circ f_1(x)$ where each $f_i(z) = \sigma_i(W_i z + b_i)$. Derive the gradient of the loss $L$ with respect to $W_1$.**
   *Answer: Using the chain rule: $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial W_1}$. Let $z_1 = W_1 x + b_1$, $a_1 = \sigma_1(z_1)$, and recursively $z_i = W_i a_{i-1} + b_i$, $a_i = \sigma_i(z_i)$. Then $\frac{\partial L}{\partial W_1} = \delta_2 \cdot W_2^T \cdots \delta_L \cdot W_L^T \cdot \sigma_1'(z_1) \cdot x^T$ where $\delta_i$ is the error signal at layer $i$.*

3. **Consider the functional equation $f(f(x)) = x$. Such functions are called involutions. Find three distinct examples and show they satisfy this equation.**
   *Answer: An involution satisfies $f \circ f = id$, meaning $f$ is its own inverse. Example 1: $f(x) = -x$. $f(f(x)) = -(-x) = x$. Example 2: $f(x) = \frac{1}{x}$ (domain $x \neq 0$). $f(f(x)) = \frac{1}{1/x} = x$. Example 3: $f(x) = a - x$ for any constant $a$. $f(f(x)) = a - (a - x) = x$. Since $f \circ f = id$, composing $f$ with itself gives the identity, so $f^{-1} = f$.*

## Practice Problems

### Easy

1. Given $f(x) = 3x - 1$ and $g(x) = x^2$, find $(f \circ g)(2)$ and $(g \circ f)(2)$.

2. Given $f(x) = \sqrt{x}$ and $g(x) = 2x + 1$, find $(f \circ g)(x)$ and its domain.

3. Given $f(x) = x + 3$ and $g(x) = 4 - x$, find $(f \circ g)(x)$ and $(g \circ f)(x)$.

4. Decompose $h(x) = \ln(3x + 2)$ into $f \circ g$.

5. If $f(x) = 2x$ and $g(x) = \frac{x}{2}$, verify that $f \circ g = g \circ f = id$.

### Medium

1. Given $f(x) = \frac{1}{x-1}$ and $g(x) = \sqrt{x}$, find $(f \circ g)(x)$ and its domain.

2. Find $f$ and $g$ such that $h(x) = \sin^2(3x)$ (i.e., $(\sin(3x))^2$).

3. Given $f(x) = x^2 - 1$ and $g(x) = \sqrt{x + 3}$, find $(g \circ f)(x)$ and its domain.

4. Use the chain rule to find $h'(x)$ where $h(x) = \cos(2x^3 - x)$.

5. If $f(x) = 2x + 5$ and $g(x) = 3x - 2$, solve $(f \circ g)(x) = (g \circ f)(x)$.

### Hard

1. Prove that if $f$ and $g$ are surjective, then $f \circ g$ is surjective.

2. For $f(x) = \frac{x}{x-1}$, find $f \circ f$, $f \circ f \circ f$, and generalize $f^{\circ n}$ ($f$ composed with itself $n$ times).

3. In a normalizing flow, $f = f_K \circ \cdots \circ f_1$ is a composition of invertible functions. Show that $\log|\det J_f(x)| = \sum_{i=1}^K \log|\det J_{f_i}(f_{i-1} \circ \cdots \circ f_1(x))|$.

## Solutions

### Easy Solutions

**1.** $f(g(2)) = f(4) = 3(4) - 1 = 11$. $g(f(2)) = g(5) = 5^2 = 25$.

**2.** $(f \circ g)(x) = \sqrt{2x + 1}$. Domain: $2x + 1 \geq 0 \implies x \geq -\frac{1}{2}$. So $[-\frac{1}{2}, \infty)$.

**3.** $(f \circ g)(x) = f(4 - x) = (4 - x) + 3 = 7 - x$. $(g \circ f)(x) = g(x + 3) = 4 - (x + 3) = 1 - x$.

**4.** Let $g(x) = 3x + 2$ and $f(u) = \ln(u)$. Then $h(x) = f(g(x)) = \ln(3x + 2)$.

**5.** $f(g(x)) = 2 \cdot \frac{x}{2} = x$. $g(f(x)) = \frac{2x}{2} = x$. Both equal $id(x) = x$. Verified.

### Medium Solutions

**1.** $(f \circ g)(x) = \frac{1}{\sqrt{x} - 1}$. Domain: $x \geq 0$ and $x \neq 1$. So $[0, 1) \cup (1, \infty)$.

**2.** $h(x) = (\sin(3x))^2$. Let $g(x) = \sin(3x)$ and $f(u) = u^2$. Then $h = f \circ g$.

**3.** $(g \circ f)(x) = \sqrt{(x^2 - 1) + 3} = \sqrt{x^2 + 2}$. Domain: $\mathbb{R}$ (since $x^2 + 2 \geq 2 > 0$ for all $x$).

**4.** $h(x) = \cos(2x^3 - x)$. $g(x) = 2x^3 - x$, $f(u) = \cos(u)$. $g'(x) = 6x^2 - 1$, $f'(u) = -\sin(u)$. $h'(x) = -\sin(2x^3 - x) \cdot (6x^2 - 1)$.

**5.** $(f \circ g)(x) = 2(3x - 2) + 5 = 6x + 1$. $(g \circ f)(x) = 3(2x + 5) - 2 = 6x + 13$. Set equal: $6x + 1 = 6x + 13 \implies 1 = 13$, impossible. No solution.

### Hard Solutions

**1.** Let $f: B \to C$ and $g: A \to B$ be surjective. Take any $c \in C$. Since $f$ is surjective, $\exists b \in B$ with $f(b) = c$. Since $g$ is surjective, $\exists a \in A$ with $g(a) = b$. Then $(f \circ g)(a) = f(g(a)) = f(b) = c$. So every $c \in C$ has a preimage under $f \circ g$. Therefore $f \circ g$ is surjective.

**2.** $f(x) = \frac{x}{x-1}$. $f \circ f(x) = \frac{\frac{x}{x-1}}{\frac{x}{x-1} - 1} = \frac{\frac{x}{x-1}}{\frac{1}{x-1}} = x$. So $f \circ f = id$. Therefore $f$ is an involution: $f^{\circ 2} = id$, $f^{\circ 3} = f$, $f^{\circ 4} = id$, etc. In general, $f^{\circ n}(x) = f(x)$ if $n$ is odd and $x$ if $n$ is even.

**3.** By the chain rule for Jacobians: $J_f(x) = J_{f_K}(z_{K-1}) \cdot J_{f_{K-1}}(z_{K-2}) \cdots J_{f_1}(x)$, where $z_i = f_i \circ \cdots \circ f_1(x)$. The determinant of a product is the product of determinants: $\det J_f(x) = \det J_{f_K}(z_{K-1}) \cdot \det J_{f_{K-1}}(z_{K-2}) \cdots \det J_{f_1}(x)$. Taking absolute value and log: $\log|\det J_f(x)| = \sum_{i=1}^K \log|\det J_{f_i}(z_{i-1})|$, where $z_0 = x$ and $z_i = f_i(z_{i-1})$.

## Related Concepts

- **Function** (MATH-044) — The basic building block; composition builds complex functions from simple ones.
- **Domain** (MATH-045) — The domain of $f \circ g$ depends on both $f$ and $g$ domains.
- **Range** (MATH-046) — The range of $g$ must intersect the domain of $f$ for $f \circ g$ to be defined.
- **Inverse Function** (MATH-048) — $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$; inverses undo composition in reverse order.
- **Chain Rule** — The derivative of a composition; the foundation of backpropagation.
- **Higher-Order Functions** — In programming, functions that take functions as arguments or return functions.
- **Category Theory** — The abstract study of composition; categories are defined by objects and composable morphisms.

## Next Concepts

- **Inverse Function** (MATH-048) — Functions that undo each other, intimately related to composition since $f^{-1} \circ f = id$.
- **Derivatives and Backpropagation** — The chain rule for compositions is the core of neural network training.
- **Function Spaces** — Sets of functions with composition as an operation.
- **Normalizing Flows** — Deep generative models built from compositions of invertible functions.

## Summary

A composite function $(f \circ g)(x) = f(g(x))$ applies $g$ first and then $f$, creating a new function from two existing ones. Composition is non-commutative ($f \circ g \neq g \circ f$ in general) but associative. The domain of $f \circ g$ requires $x$ to be in the domain of $g$ and $g(x)$ to be in the domain of $f$. Composition is central to deep learning: neural networks are hierarchical compositions of affine transformations and non-linear activations. The chain rule for compositions is the mathematical foundation of backpropagation, the algorithm used to train neural networks. Understanding composition enables decomposing complex functions into simpler parts, building ML pipelines, and designing architectures like ResNets and normalizing flows.

## Key Takeaways

- $(f \circ g)(x) = f(g(x))$ — apply $g$ first, then $f$. The order matters.
- Composition is not commutative: $f \circ g \neq g \circ f$ generally.
- $\text{dom}(f \circ g) = \{x \in \text{dom}(g) : g(x) \in \text{dom}(f)\}$.
- Composition is associative: $(f \circ g) \circ h = f \circ (g \circ h)$.
- Composition preserves injectivity, surjectivity, and bijectivity.
- The chain rule $(f \circ g)'(x) = f'(g(x)) g'(x)$ is the foundation of backpropagation.
- Deep neural networks are compositions of layer functions: $f_L \circ \cdots \circ f_1$.
- Decomposition (breaking a complex function into simpler parts) is a key problem-solving skill.
- The inverse of a composition reverses order: $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$.
- ML pipelines (e.g., scikit-learn Pipeline) implement function composition in software.
