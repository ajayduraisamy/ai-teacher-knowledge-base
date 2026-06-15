# Concept: Real Numbers

## Concept ID

MATH-008

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Foundations

## Learning Objectives

- Define the set of real numbers $\mathbb{R}$ and distinguish its subsets
- Understand and apply the field properties: closure, commutativity, associativity, distributivity
- Explain the density property and completeness of real numbers
- Work with absolute value and its properties
- Relate real number properties to machine learning concepts

## Prerequisites

- Familiarity with number systems ($\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$) from MATH-007
- Basic arithmetic operations
- Understanding of fractions and decimals

## Definition

The **real numbers** $\mathbb{R}$ form the set of all numbers that can be represented as points on an infinite continuous line called the **real number line**. Every real number is either rational (expressible as $\frac{a}{b}$ with $a, b \in \mathbb{Z}$, $b \neq 0$) or irrational (non-terminating, non-repeating decimal). The real numbers include:

- Natural numbers: $1, 2, 3, \dots$
- Whole numbers: $0, 1, 2, 3, \dots$
- Integers: $\dots, -2, -1, 0, 1, 2, \dots$
- Rational numbers: $\frac{1}{2}, -\frac{3}{4}, 0.333..., 0.25, \dots$
- Irrational numbers: $\sqrt{2}, \pi, e, \dots$

## Intuition

Imagine drawing an infinitely long straight line. Mark a point as 0. Pick a unit length and mark 1, 2, 3, ... to the right and -1, -2, -3, ... to the left. Now imagine every possible point on this line having a number — not just the tick marks, but every location including those between, no matter how finely you zoom in. That is the real number line. It has **no gaps** — this property is called **completeness**.

Real numbers are what we use in everyday measurement: lengths, weights, temperatures, times. Any quantity that can vary continuously is modeled using real numbers.

## Why This Concept Matters

Real numbers are the fundamental numeric domain for calculus, analysis, and most of applied mathematics. They are also the default numeric type in machine learning — every weight, bias, feature value, and gradient in a neural network is a real number. Understanding the properties of real numbers (especially density, completeness, and absolute value) is essential for:

- Understanding limits and continuity (foundation of calculus and optimization)
- Convergence of training algorithms (gradient descent, SGD)
- Numerical stability and error analysis
- Distance metrics and regularization in ML

## Historical Background

The concept of real numbers evolved over millennia:

- **Ancient Greeks (c. 500 BCE)**: Pythagoreans believed all numbers were rational until Hippasus proved $\sqrt{2}$ was irrational. This caused a foundational crisis.
- **Eudoxus (c. 400 BCE)**: Developed a theory of proportions that handled incommensurable quantities without explicitly naming irrational numbers.
- **Middle Ages**: Islamic mathematicians like Al-Khwarizmi worked with irrational numbers in algebra.
- **16th–17th centuries**: European mathematicians (Stevin, Descartes, Newton) freely used decimals and irrationals without rigorous foundations.
- **19th century**: Rigorous foundations were laid. **Richard Dedekind** (1872) defined reals via "Dedekind cuts" — partitioning $\mathbb{Q}$ into two sets. **Georg Cantor** defined reals via Cauchy sequences of rationals. **Karl Weierstrass** developed the epsilon-delta definition of limits.
- **20th century**: The real numbers were axiomatized as a **complete ordered field**, unique up to isomorphism.

## Real World Examples

1. **Temperature**: Thermometer readings are real numbers (e.g., 23.5°C, -4.2°C). Temperature varies continuously.
2. **Distance**: The distance between two cities (e.g., 152.7 km) is a real number. So is the diagonal of a 1m×1m square: $\sqrt{2} \approx 1.4142$ m.
3. **Time**: Elapsed time is a real number (e.g., 3.14159 seconds). Time is continuous.
4. **Stock Prices**: A stock price of 147.83 dollars is a real number (though in practice it's rational as currency is discrete to cents).
5. **Probability**: The probability of an event is a real number between 0 and 1 inclusive, e.g., $P(\text{heads}) = 0.5$, $P(\text{rain}) = 0.333...$.

## AI/ML Relevance

- **Continuous features**: In ML, most numerical features (age, salary, temperature, pixel intensity) are modeled as real numbers. Algorithms like linear regression, neural networks, and SVMs assume real-valued inputs.
- **Loss functions**: Loss functions are real-valued functions of real parameters. Mean Squared Error: $L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$. Cross-entropy: $L = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i)]$.
- **Gradient descent**: Updates weights by subtracting a fraction of the gradient: $w_{t+1} = w_t - \eta \nabla L(w_t)$. This relies on the real numbers being closed under addition and scalar multiplication.
- **Distance metrics**: Euclidean distance between two $d$-dimensional points $a, b \in \mathbb{R}^d$: $||a - b||_2 = \sqrt{\sum_{i=1}^d (a_i - b_i)^2}$. This uses square roots (irrational operations) on reals.
- **Regularization**: L1 regularization (Lasso) uses absolute value: $R = \lambda \sum |w_i|$. L2 regularization (Ridge) uses squares: $R = \lambda \sum w_i^2$.
- **Activation functions**: Sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$, ReLU $\max(0, x)$, tanh — all map real numbers to real numbers.
- **Numerical precision**: Computers approximate $\mathbb{R}$ with floating-point numbers (finite subset of $\mathbb{Q}$). This discretization causes rounding errors, which can accumulate during training.
- **Batch normalization**: Normalizes activations to have zero mean and unit variance — operations defined on real numbers.

## Mathematical Explanation

### Subsets of Real Numbers

$$\mathbb{R} = \mathbb{Q} \cup \mathbb{Q}'$$

Where $\mathbb{Q}$ are rationals and $\mathbb{Q}'$ are irrationals.

$$\mathbb{Q} = \left\{\frac{a}{b} \mid a, b \in \mathbb{Z},\; b \neq 0\right\}$$

Irrational numbers cannot be expressed as a ratio of integers. Their decimal expansions are non-terminating and non-repeating.

### The Real Number Line

Every real number corresponds to exactly one point on the number line, and vice versa. This is called the **continuum** property. The number line is:

- **Infinite** in both directions
- **Dense**: between any two distinct real numbers, there is another real number
- **Complete**: no gaps (every Cauchy sequence converges)

### Order Properties

Real numbers are **ordered**: for any $a, b \in \mathbb{R}$, exactly one of $a < b$, $a = b$, or $a > b$ holds. Key order properties:

- **Transitivity**: If $a < b$ and $b < c$, then $a < c$.
- **Addition preserves order**: If $a < b$, then $a + c < b + c$.
- **Multiplication by positives preserves order**: If $a < b$ and $c > 0$, then $ac < bc$.
- **Multiplication by negatives reverses order**: If $a < b$ and $c < 0$, then $ac > bc$.

### Absolute Value

The **absolute value** of a real number $x$, denoted $|x|$, is its distance from zero on the number line:

$$|x| = \begin{cases}
x & \text{if } x \geq 0 \\
-x & \text{if } x < 0
\end{cases}$$

**Properties of absolute value**:
1. $|x| \geq 0$ for all $x$, and $|x| = 0$ iff $x = 0$.
2. $|xy| = |x| \cdot |y|$.
3. $\left|\frac{x}{y}\right| = \frac{|x|}{|y|}$ for $y \neq 0$.
4. **Triangle inequality**: $|x + y| \leq |x| + |y|$.
5. **Reverse triangle inequality**: $||x| - |y|| \leq |x - y|$.

### Density of Real Numbers

Between any two distinct real numbers $a < b$, there exists:

- A rational number $q \in \mathbb{Q}$ such that $a < q < b$.
- An irrational number $r \notin \mathbb{Q}$ such that $a < r < b$.

This means the rationals and irrationals are both "everywhere" on the number line — no matter how small an interval you pick, it contains both types.

### Completeness

The **completeness axiom** (or least upper bound property) states: every non-empty set of real numbers that is bounded above has a **least upper bound** (supremum) in $\mathbb{R}$.

This is what distinguishes $\mathbb{R}$ from $\mathbb{Q}$. For example, the set $\{x \in \mathbb{Q} \mid x^2 < 2\}$ is bounded above (by 2, for instance), but it has no rational supremum (since $\sqrt{2} \notin \mathbb{Q}$). In $\mathbb{R}$, the supremum is $\sqrt{2}$.

## Formula(s)

- **Absolute value definition**: $|x| = \begin{cases} x & x \geq 0 \\ -x & x < 0 \end{cases}$
- **Triangle inequality**: $|x + y| \leq |x| + |y|$
- **Reverse triangle inequality**: $||x| - |y|| \leq |x - y|$
- **Distance between two points**: $d(a, b) = |a - b|$
- **Euclidean norm in $\mathbb{R}^n$**: $||\mathbf{x}||_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$
- **Arithmetic mean**: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$
- **Variance**: $\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2$

## Properties

### Field Properties

Real numbers form a **field** under addition and multiplication, satisfying:

| Property | Addition | Multiplication |
|----------|----------|---------------|
| **Closure** | $a + b \in \mathbb{R}$ | $ab \in \mathbb{R}$ |
| **Commutativity** | $a + b = b + a$ | $ab = ba$ |
| **Associativity** | $(a + b) + c = a + (b + c)$ | $(ab)c = a(bc)$ |
| **Identity** | $a + 0 = a$ | $a \cdot 1 = a$ |
| **Inverse** | $a + (-a) = 0$ | $a \cdot a^{-1} = 1$ ($a \neq 0$) |
| **Distributivity** | $a(b + c) = ab + ac$ | — |

### Order Properties

- Real numbers are **totally ordered** (trichotomy law).
- The order is compatible with addition and multiplication (as described above).

### Completeness

- Every Cauchy sequence of real numbers converges to a real number.
- Every non-empty set bounded above has a supremum in $\mathbb{R}$.
- Every non-empty set bounded below has an infimum in $\mathbb{R}$.

### Archimedean Property

For any real number $x$, there exists a natural number $n$ such that $n > x$. Equivalently, for any positive real $\epsilon$, there exists $n \in \mathbb{N}$ such that $\frac{1}{n} < \epsilon$.

### Density

Both $\mathbb{Q}$ and $\mathbb{Q}'$ (irrationals) are dense in $\mathbb{R}$.

## Step-by-Step Worked Examples

### Example 1: Applying Field Properties

**Problem**: Identify which field property justifies each statement.

**(a)** $3 + (5 + 2) = (3 + 5) + 2$
**Step 1**: The grouping of numbers (parentheses) changes without changing the sum.
**Answer**: Associativity of addition.

**(b)** $7 \cdot (3 + 4) = 7 \cdot 3 + 7 \cdot 4$
**Step 1**: Multiplication distributes over addition.
**Answer**: Distributive property.

**(c)** $-8 + 8 = 0$
**Step 1**: Adding a number and its additive inverse yields the identity element 0.
**Answer**: Additive inverse property.

**(d)** $3.14 \times 0 = 0$
**Step 1**: Any real number times zero equals zero (follows from distributivity: $a \cdot 0 = a(0+0) = a\cdot 0 + a\cdot 0 \Rightarrow a \cdot 0 = 0$).
**Answer**: Zero multiplication property (derived from distributivity).

### Example 2: Absolute Value Equations

**Problem**: Solve $|x - 3| = 5$.

**Step 1**: Recall that $|u| = c$ means $u = c$ or $u = -c$ (for $c > 0$).
$$x - 3 = 5 \quad \text{or} \quad x - 3 = -5$$

**Step 2**: Solve each equation.
$$x = 8 \quad \text{or} \quad x = -2$$

**Step 3**: Verify.
$|8 - 3| = |5| = 5$, $|-2 - 3| = |-5| = 5$.

**Answer**: $x = 8$ or $x = -2$.

### Example 3: Triangle Inequality

**Problem**: Show that $|x + y| \leq |x| + |y|$ for $x = -3$ and $y = 5$.

**Step 1**: Compute the left side.
$$|x + y| = |-3 + 5| = |2| = 2$$

**Step 2**: Compute the right side.
$$|x| + |y| = |-3| + |5| = 3 + 5 = 8$$

**Step 3**: Compare.
$$2 \leq 8$$

**Answer**: The triangle inequality holds: $2 \leq 8$.

### Example 4: Density of Rationals

**Problem**: Find a rational number between $\sqrt{2} \approx 1.41421356...$ and $\sqrt{3} \approx 1.73205080...$.

**Step 1**: Compute the midpoint.
$$\frac{\sqrt{2} + \sqrt{3}}{2} \approx \frac{1.4142 + 1.7321}{2} = 1.57315$$

**Step 2**: This number is approximately 1.57315. But is it rational? Not necessarily.

**Step 3**: Use the Archimedean property to guarantee a rational. Let $x = \sqrt{2}$, $y = \sqrt{3}$. Choose $n$ such that $\frac{1}{n} < y - x \approx 0.3178$, e.g., $n = 4$ since $\frac{1}{4} = 0.25 < 0.3178$.

**Step 4**: Let $m = \lceil n x \rceil = \lceil 4 \times 1.4142 \rceil = \lceil 5.6568 \rceil = 6$.

**Step 5**: Then $\frac{m}{n} = \frac{6}{4} = 1.5$, and $\sqrt{2} < 1.5 < \sqrt{3}$.

**Answer**: $\frac{3}{2} = 1.5$ is a rational number between $\sqrt{2}$ and $\sqrt{3}$.

### Example 5: Solving an Inequality with Absolute Value

**Problem**: Solve $|2x - 1| < 3$.

**Step 1**: $|u| < c$ means $-c < u < c$ (for $c > 0$).
$$-3 < 2x - 1 < 3$$

**Step 2**: Add 1 to all three parts.
$$-2 < 2x < 4$$

**Step 3**: Divide by 2.
$$-1 < x < 2$$

**Step 4**: Check a value: $x = 0$ gives $|2(0) - 1| = |-1| = 1 < 3$.

**Answer**: $x \in (-1, 2)$.

### Example 6: Completeness (Least Upper Bound)

**Problem**: Find the supremum (least upper bound) of $S = \{x \in \mathbb{R} \mid x^2 < 4\}$.

**Step 1**: $S$ consists of all real numbers whose square is less than 4. This is the open interval $(-2, 2)$.

**Step 2**: Upper bounds of $S$: any number $\geq 2$.

**Step 3**: The least upper bound is 2. Is $2 \in S$? $2^2 = 4 \not< 4$, so $2 \notin S$.

**Step 4**: The supremum is 2 (even though the maximum does not exist, since 2 is not in $S$).

**Answer**: $\sup(S) = 2$.

## Visual Interpretation

- **Number line**: Real numbers fill the entire continuous line. No matter how far you zoom in, there are always more numbers. Rationals are like dust particles densely scattered everywhere; irrationals fill the remaining gaps.
- **Absolute value as distance**: $|a - b|$ is the distance between $a$ and $b$ on the number line. $|x|$ is the distance from $x$ to 0.
- **Completeness**: Imagine the rational numbers on a line with tiny holes at every irrational. The real numbers fill those holes, making the line smooth and continuous. If you walk along the real line, you never need to jump.
- **Triangle inequality**: In a triangle formed by vectors $x$ and $y$, the direct path (sum) is always $\leq$ the sum of the two sides. This is why "shortcuts" are shorter or equal.

## Common Mistakes

1. **Confusing supremum with maximum**: The supremum is the least upper bound; it may or may not belong to the set. The maximum must belong to the set. $S = (0, 1)$ has supremum 1 but no maximum.
2. **Thinking all real numbers are rational**: $\sqrt{2}$, $\pi$, $e$ are real but not rational. An irrational number cannot be written as a simple fraction.
3. **Forgetting that absolute value is always non-negative**: $|x|$ is distance from zero; it is never negative. A common error is writing $|x| = \pm x$.
4. **Misapplying the triangle inequality**: $|x + y| \leq |x| + |y|$, not $|x + y| = |x| + |y|$. Equality holds only when $x$ and $y$ have the same sign (or one is zero).
5. **Assuming $\mathbb{R}$ is countable**: The reals are uncountably infinite — there are more real numbers between 0 and 1 than there are integers in total.
6. **Dividing by a variable when solving inequalities**: If you multiply or divide by a negative number, the inequality sign flips. Forgetting this leads to incorrect solutions.
7. **Believing $|x| = -x$ means $x$ is negative**: $|x| = -x$ when $x \leq 0$. If $x = -3$, $| -3| = 3 = -(-3)$. The expression $-x$ when $x$ is negative is actually positive.

## Interview Questions

### Beginner

**Q1**: What is a real number?
**A**: A real number is any number that can be represented as a point on the continuous number line. This includes rational numbers (fractions and terminating/repeating decimals) and irrational numbers (non-terminating, non-repeating decimals like $\pi$ and $\sqrt{2}$).

**Q2**: What does absolute value mean?
**A**: The absolute value of a number $x$, written $|x|$, is the distance of $x$ from 0 on the number line. It is always non-negative. For example, $|5| = 5$ and $|-5| = 5$.

**Q3**: What is the commutative property of addition?
**A**: The commutative property states that the order of addition does not matter: $a + b = b + a$ for all real numbers $a$ and $b$.

**Q4**: Is $\pi$ a real number? Is it rational?
**A**: $\pi$ is a real number (it lies on the number line at approximately 3.14159), but it is irrational — its decimal expansion never terminates or repeats, and it cannot be expressed as a fraction.

**Q5**: What is the distributive property?
**A**: The distributive property connects addition and multiplication: $a(b + c) = ab + ac$ for all real numbers $a$, $b$, $c$. For example, $2(3 + 4) = 2 \cdot 7 = 14$ and $2 \cdot 3 + 2 \cdot 4 = 6 + 8 = 14$.

### Intermediate

**Q1**: State and explain the triangle inequality.
**A**: The triangle inequality states $|x + y| \leq |x| + |y|$ for all real numbers $x, y$. It says the absolute value of a sum is at most the sum of the absolute values. Equality holds when $x$ and $y$ have the same sign. It is called the triangle inequality because in a triangle, one side length is at most the sum of the other two.

**Q2**: What is the difference between the supremum and the maximum of a set?
**A**: The supremum (least upper bound) is the smallest number that is $\geq$ every element of the set. The maximum is the largest element that actually belongs to the set. A set like $(0, 1)$ has supremum 1 but no maximum (since 1 is not in the set). A set like $[0, 1]$ has both supremum and maximum equal to 1.

**Q3**: Explain the completeness property of real numbers.
**A**: Completeness means there are no "gaps" in the real numbers. Formally, every non-empty set of real numbers that is bounded above has a least upper bound (supremum) in $\mathbb{R}$. This property distinguishes $\mathbb{R}$ from $\mathbb{Q}$ — the set $\{x \in \mathbb{Q} \mid x^2 < 2\}$ has no rational supremum (since $\sqrt{2} \notin \mathbb{Q}$), but in $\mathbb{R}$, it does.

**Q4**: Why is the set of rational numbers not complete?
**A**: $\mathbb{Q}$ is not complete because we can find a bounded sequence of rationals whose limit is irrational. For example, the sequence $1, 1.4, 1.41, 1.414, 1.4142, \dots$ (rational approximations of $\sqrt{2}$) converges to $\sqrt{2}$, which is not in $\mathbb{Q}$. The supremum of $\{x \in \mathbb{Q} \mid x^2 < 2\}$ does not exist in $\mathbb{Q}$.

**Q5**: Solve $|2x + 3| \geq 7$ and express the solution in interval notation.
**A**: $|2x + 3| \geq 7$ means $2x + 3 \leq -7$ or $2x + 3 \geq 7$. For the first: $2x \leq -10 \Rightarrow x \leq -5$. For the second: $2x \geq 4 \Rightarrow x \geq 2$. Solution: $(-\infty, -5] \cup [2, \infty)$.

### Advanced

**Q1**: Prove that between any two distinct real numbers, there exists an irrational number.
**A**: Let $a < b$ be real numbers. We know there exists a rational $q$ between $a$ and $b$ (density of $\mathbb{Q}$). Consider $r = q + \frac{\sqrt{2}}{n}$ where $n$ is chosen large enough so that $r < b$ — specifically, $n > \frac{\sqrt{2}}{b - q}$. Since adding an irrational ($\sqrt{2}/n$ is irrational because $\sqrt{2}$ is irrational and $n$ is rational) to a rational yields an irrational, $r$ is irrational and $a < q < r < b$.

**Q2**: Prove the Archimedean property: for any real $x > 0$, there exists $n \in \mathbb{N}$ such that $\frac{1}{n} < x$.
**A**: Assume the contrary — that $\frac{1}{n} \geq x$ for all $n \in \mathbb{N}$, i.e., $n \leq \frac{1}{x}$ for all $n$. This would mean $\mathbb{N}$ is bounded above by $\frac{1}{x}$, contradicting the completeness axiom (a non-empty set bounded above must have a supremum, but $\mathbb{N}$ has no finite supremum). Therefore, such an $n$ must exist.

**Q3**: Explain how the completeness of $\mathbb{R}$ ensures the convergence of gradient descent in machine learning.
**A**: Gradient descent produces a sequence $\{w_t\}$ of parameter vectors. Under mild conditions (convexity, Lipschitz gradient), the sequence is bounded and monotone in the value of the loss function. The completeness of $\mathbb{R}$ guarantees that such a bounded monotone sequence converges to a limit point $w^*$. Without completeness, the sequence could "want" to converge to a limit that does not exist in the space (like a sequence of rationals converging to $\sqrt{2}$). Completeness ensures that limits of convergent sequences exist within $\mathbb{R}$, guaranteeing that gradient descent reaches a well-defined solution.

## Practice Problems

### Easy - 5 Questions

**E1**: Simplify $| -7 | + | 3 |$.
**E2**: Which property is illustrated by $(2 \times 3) \times 4 = 2 \times (3 \times 4)$?
**E3**: Is $0.333...$ rational or irrational? Explain.
**E4**: Find the distance between $-5$ and $3$ on the number line.
**E5**: Solve $|x| = 6$.

### Medium - 5 Questions

**M1**: Solve $|2x - 5| \leq 9$.
**M2**: Find the supremum and maximum (if they exist) of $S = \{x \in \mathbb{R} \mid 0 \leq x < 5\}$.
**M3**: Prove that $\sqrt{5}$ is irrational.
**M4**: Find two rational numbers between $\pi$ and $3.2$.
**M5**: Show that $|a - b| \geq ||a| - |b||$ for $a = 7$ and $b = -3$ (reverse triangle inequality).

### Hard - 3 Questions

**H1**: Prove that $\sqrt{2} + \sqrt{3}$ is irrational.
**H2**: Using the completeness axiom, prove that every non-empty set of real numbers bounded below has a greatest lower bound (infimum).
**H3**: Let $f(x) = w^T x + b$ be a linear model where $w \in \mathbb{R}^d$, $x \in \mathbb{R}^d$, $b \in \mathbb{R}$. Show that the set of outputs $\{f(x) \mid x \in \mathbb{R}^d, ||x||_2 \leq 1\}$ is bounded and find its supremum in terms of $w$ and $b$.

## Solutions

### Easy Solutions

**E1**: $|-7| + |3| = 7 + 3 = 10$.

**E2**: Associativity of multiplication. The grouping of factors changes without affecting the product.

**E3**: $0.333...$ is rational because it is a repeating decimal: $0.333... = \frac{1}{3}$, a ratio of integers.

**E4**: Distance $= |3 - (-5)| = |8| = 8$ units.

**E5**: $|x| = 6$ means $x = 6$ or $x = -6$.

### Medium Solutions

**M1**: $|2x - 5| \leq 9$ means $-9 \leq 2x - 5 \leq 9$. Add 5: $-4 \leq 2x \leq 14$. Divide by 2: $-2 \leq x \leq 7$. Solution: $x \in [-2, 7]$.

**M2**: $S = [0, 5)$. Upper bounds: all numbers $\geq 5$. Least upper bound (supremum) = 5. Maximum: 5 is not in $S$ (since $x < 5$), so no maximum exists. Infimum = 0, minimum = 0.

**M3**: Assume $\sqrt{5} = a/b$ in lowest terms. Then $5b^2 = a^2$. So $a^2$ is divisible by 5, hence $a$ is divisible by 5: $a = 5k$. Then $5b^2 = 25k^2 \Rightarrow b^2 = 5k^2$, so $b$ is also divisible by 5. Contradiction — $a$ and $b$ share factor 5. Hence $\sqrt{5}$ is irrational.

**M4**: $\pi \approx 3.14159$, $3.2 = 3.2$. Two rational numbers: $3.15 = \frac{63}{20}$ and $3.18 = \frac{159}{50}$; both lie strictly between $\pi$ and $3.2$.

**M5**: Left side: $|7 - (-3)| = |10| = 10$. Right side: $||7| - |-3|| = |7 - 3| = |4| = 4$. Indeed $10 \geq 4$, verifying $|a - b| \geq ||a| - |b||$.

### Hard Solutions

**H1**: Let $x = \sqrt{2} + \sqrt{3}$. Square: $x^2 = 2 + 2\sqrt{6} + 3 = 5 + 2\sqrt{6}$. So $\sqrt{6} = \frac{x^2 - 5}{2}$. If $x$ were rational, then $\frac{x^2 - 5}{2}$ would be rational, implying $\sqrt{6}$ is rational. But $\sqrt{6}$ is irrational (proof by contradiction like $\sqrt{2}$). Therefore $x$ must be irrational.

**H2**: Let $S$ be a non-empty set of real numbers bounded below. Define $T = \{-x \mid x \in S\}$. $T$ is non-empty and bounded above (if $L$ is a lower bound of $S$, then $-L$ is an upper bound of $T$). By completeness, $T$ has a least upper bound $u = \sup(T)$. Since $u \geq t$ for all $t \in T$, we have $-u \leq -t = x$ for all $x \in S$, so $-u$ is a lower bound of $S$. If $v > -u$ were a greater lower bound, then $-v < u$ would be an upper bound of $T$ smaller than $u$, contradicting $u = \sup(T)$. Therefore $-u = \inf(S)$.

**H3**: For any $x$ with $||x||_2 \leq 1$, by Cauchy-Schwarz inequality: $|w^T x| \leq ||w||_2 \cdot ||x||_2 \leq ||w||_2 \cdot 1 = ||w||_2$. Therefore $w^T x \in [-||w||_2, ||w||_2]$. Then $f(x) = w^T x + b \in [b - ||w||_2, b + ||w||_2]$. The set is bounded, and the supremum is $b + ||w||_2$, achieved when $x = \frac{w}{||w||_2}$ (the unit vector in the direction of $w$).

## Related Concepts

- **Rational Numbers** $\mathbb{Q}$: A subset of $\mathbb{R}$; understanding $\mathbb{Q}$ is essential for grasping what $\mathbb{R}$ adds.
- **Irrational Numbers**: The numbers in $\mathbb{R}$ that are not rational.
- **Complex Numbers** $\mathbb{C}$: Extends $\mathbb{R}$ by adding $i = \sqrt{-1}$.
- **Number Line**: The geometric representation of $\mathbb{R}$.
- **Set Theory**: Provides language for describing subsets of $\mathbb{R}$.

## Next Concepts

- **Functions**: Real-valued functions $f: \mathbb{R} \to \mathbb{R}$.
- **Limits and Continuity**: Foundational concepts of calculus built on completeness.
- **Differentiation**: Rates of change of real-valued functions.
- **Integration**: Accumulation of real quantities.
- **Metric Spaces**: Generalization of distance (absolute value) to abstract spaces.
- **Topology**: Open sets, closed sets, compactness on $\mathbb{R}$.

## Summary

The real numbers $\mathbb{R}$ form a **complete ordered field** — the continuous number line with no gaps. They include rationals and irrationals, satisfy field properties (closure, commutativity, associativity, distributivity), have a total order, and are complete (every bounded set has a supremum).

Key properties:
- **Field properties**: $\mathbb{R}$ is closed under $+$, $-$, $\times$, $\div$ (except by 0), with commutative, associative, and distributive laws.
- **Order**: For any $a, b \in \mathbb{R}$, exactly one of $a < b$, $a = b$, $a > b$ holds.
- **Absolute value**: $|x|$ measures distance from 0; satisfies the triangle inequality.
- **Density**: Between any two reals, there exist both a rational and an irrational.
- **Completeness**: Every non-empty set bounded above has a least upper bound in $\mathbb{R}$.

These properties make $\mathbb{R}$ the natural setting for calculus, optimization, and machine learning — where continuous quantities, limits, and convergence are fundamental.

## Key Takeaways

- Real numbers $\mathbb{R}$ = all points on the continuous number line = rationals $\cup$ irrationals.
- $\mathbb{R}$ satisfies the field properties: closure, commutativity, associativity, distributivity, identities, and inverses.
- Absolute value $|x|$ is distance from zero; the triangle inequality $|x+y| \leq |x| + |y|$ is fundamental.
- Density: between any two reals, there is a rational and an irrational.
- Completeness: every bounded set has a supremum in $\mathbb{R}$ — this distinguishes $\mathbb{R}$ from $\mathbb{Q}$.
- The Archimedean property guarantees arbitrarily small positive reals exist.
- In ML, real numbers are the domain of features, weights, gradients, loss functions, and activation functions.
- The completeness of $\mathbb{R}$ ensures that optimization algorithms like gradient descent converge to well-defined limits.
- Numerical computation on computers approximates $\mathbb{R}$ with finite-precision floating-point numbers, introducing rounding errors that practitioners must understand.
