# Concept: Number Systems

## Concept ID

MATH-007

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Foundations

## Learning Objectives

- Understand the hierarchy of number systems: $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$
- Explain why each number system was historically developed
- Perform operations within and between number systems
- Recognize the cardinality differences between countable and uncountable infinities
- Relate number systems to data types in computing and machine learning

## Prerequisites

- Basic arithmetic (addition, subtraction, multiplication, division)
- Understanding of fractions and decimals
- Familiarity with negative numbers

## Definition

A **number system** is a set of numbers together with operations (typically addition and multiplication) defined on them. Mathematicians organize numbers into a hierarchy of systems, each extending the previous one to solve new types of problems. The main number systems are:

- **Natural numbers** $\mathbb{N}$: $\{1, 2, 3, \dots\}$
- **Whole numbers** $\mathbb{W}$: $\{0, 1, 2, 3, \dots\}$
- **Integers** $\mathbb{Z}$: $\{\dots, -3, -2, -1, 0, 1, 2, 3, \dots\}$
- **Rational numbers** $\mathbb{Q}$: numbers expressible as $\frac{a}{b}$ where $a, b \in \mathbb{Z}$ and $b \neq 0$
- **Real numbers** $\mathbb{R}$: all rational and irrational numbers (every number on the continuous number line)
- **Complex numbers** $\mathbb{C}$: numbers of the form $a + bi$ where $a, b \in \mathbb{R}$ and $i = \sqrt{-1}$

## Intuition

Imagine building a toolkit for counting and measuring. First, you have the natural numbers $\mathbb{N}$ — perfect for counting apples: 1, 2, 3, ... But what if you have no apples? You need zero. What if you owe someone 5 apples? You need negatives — enter $\mathbb{Z}$. But what if you need half an apple? Fractions — enter $\mathbb{Q}$. But what if you need to measure the diagonal of a unit square ($\sqrt{2}$)? That's not a fraction — enter $\mathbb{R}$. Finally, what if you need to solve $x^2 = -1$? No real number works — enter $\mathbb{C}$.

Each extension solved a problem that the previous system could not handle.

## Why This Concept Matters

Number systems are the foundation of all mathematics, science, and engineering. They determine what kinds of calculations are possible in a given context. In computing, the choice of number type (integer vs. float vs. complex) directly affects precision, performance, and memory usage. In machine learning, understanding whether data is real-valued, integer-valued, or categorical informs which algorithms are appropriate. The distinction between rational and irrational numbers is at the heart of floating-point arithmetic and numerical precision issues.

## Historical Background

- **Natural numbers** have been used since prehistoric times — tally marks on bones date back 30,000 years.
- **Zero** was invented independently in several cultures. The Babylonian number system (c. 300 BCE) used a placeholder zero. The concept of zero as a number was formalized in India by **Brahmagupta** (c. 628 CE).
- **Negative numbers** were first used in Chinese mathematics (c. 200 BCE) and later formalized in Indian mathematics. European mathematicians resisted them until the 17th century, calling them "absurd" or "fictitious."
- **Rational numbers** (fractions) were used by ancient Egyptians (c. 1650 BCE) in the Rhind Mathematical Papyrus.
- **Irrational numbers** were discovered by the Pythagorean **Hippasus of Metapontum** (c. 500 BCE), who proved that $\sqrt{2}$ cannot be expressed as a fraction. Legend says this discovery was so disturbing to the Pythagoreans (who believed all numbers were rational) that they drowned Hippasus.
- **Real numbers** were rigorously formalized in the 19th century by **Richard Dedekind** (Dedekind cuts) and **Georg Cantor** (Cauchy sequences).
- **Complex numbers** emerged in the 16th century from the work of **Gerolamo Cardano** on cubic equations. **Rafael Bombelli** developed rules for manipulating them. **Leonhard Euler** introduced the notation $i = \sqrt{-1}$ in the 18th century. **Carl Friedrich Gauss** formally established complex numbers in the 19th century.

## Real World Examples

1. **Banking**: Integers track deposits and withdrawals (positive and negative). Rational numbers (to two decimal places) calculate interest.
2. **Measurement**: A carpenter measures lengths in fractions of an inch ($\mathbb{Q}$). A physicist uses real numbers for continuous quantities.
3. **Digital Audio**: Sound waves are sampled and quantized. Each sample is stored as an integer (e.g., 16-bit signed integer in $\mathbb{Z}$ between -32768 and 32767).
4. **Image Processing**: Pixel intensities are integers (0–255 for 8-bit images). Image transformations use real-number arithmetic.
5. **Electrical Engineering**: Complex numbers $\mathbb{C}$ are essential for analyzing AC circuits, impedance, and signal processing.

## AI/ML Relevance

- **Feature types**: Numerical features in ML can be discrete (integers $\mathbb{Z}$) or continuous (real numbers $\mathbb{R}$). The type determines the choice of model and preprocessing.
- **Loss functions**: Mean Squared Error (MSE) operates on real numbers $\mathbb{R}$. Cross-entropy loss uses logarithms, which require positive real arguments.
- **Gradients**: Gradient descent requires real-valued gradients. Integer weights would make differentiation impossible — hence neural networks use floating-point reals.
- **Embeddings**: Word embeddings (e.g., Word2Vec, GloVe) represent words as vectors of real numbers in $\mathbb{R}^d$. Each dimension is a real-valued coordinate.
- **Complex-valued neural networks**: Some architectures (CVNNs) use complex numbers $\mathbb{C}$ for tasks involving phase information, such as radar signal processing and audio analysis.
- **Numerical precision**: Floating-point numbers (IEEE 754) approximate $\mathbb{R}$ using a finite set of rational numbers. Understanding $\mathbb{Q}$ vs. $\mathbb{R}$ helps explain rounding errors in training.
- **Quantization**: Converting real-number model weights to lower-precision integers (e.g., INT8) for deployment on edge devices is a practical mapping between $\mathbb{R}$ and $\mathbb{Z}$.

## Mathematical Explanation

### Natural Numbers $\mathbb{N}$

$$\mathbb{N} = \{1, 2, 3, 4, \dots\}$$

Some definitions include 0. In this text, $\mathbb{N}$ starts at 1. The set with 0 is called **whole numbers** $\mathbb{W} = \{0, 1, 2, 3, \dots\}$.

**Properties**: 
- Closed under addition and multiplication
- Not closed under subtraction ($2 - 5 \notin \mathbb{N}$) or division ($3 \div 2 \notin \mathbb{N}$)

### Integers $\mathbb{Z}$

$$\mathbb{Z} = \{\dots, -3, -2, -1, 0, 1, 2, 3, \dots\}$$

The letter $\mathbb{Z}$ comes from the German word *Zahlen* meaning "numbers."

**Properties**:
- Closed under addition, multiplication, and subtraction
- Not closed under division ($1 \div 2 \notin \mathbb{Z}$)

### Rational Numbers $\mathbb{Q}$

$$\mathbb{Q} = \left\{\frac{a}{b} \mid a, b \in \mathbb{Z},\; b \neq 0\right\}$$

The letter $\mathbb{Q}$ stands for "quotient." Every rational number has a decimal representation that either terminates (like $1/4 = 0.25$) or repeats (like $1/3 = 0.333...$).

**Properties**:
- Closed under addition, multiplication, subtraction, and division (except by zero)
- Dense in $\mathbb{R}$: between any two rational numbers, there is another rational number
- Countably infinite (can be listed in a sequence)

### Irrational Numbers

Numbers that cannot be expressed as $\frac{a}{b}$ with $a, b \in \mathbb{Z}$. Their decimal expansions are non-terminating and non-repeating.

Examples: $\sqrt{2} = 1.41421356\dots$, $\pi = 3.14159265\dots$, $e = 2.71828182\dots$

### Real Numbers $\mathbb{R}$

$$\mathbb{R} = \mathbb{Q} \cup \{\text{irrational numbers}\}$$

The real numbers fill the continuous number line with no gaps. The set of real numbers is **uncountably infinite** — its cardinality is strictly larger than that of $\mathbb{Q}$.

### Complex Numbers $\mathbb{C}$

$$\mathbb{C} = \{a + bi \mid a, b \in \mathbb{R},\; i^2 = -1\}$$

$i$ is the **imaginary unit**. $a$ is the **real part** and $b$ is the **imaginary part**.

**Properties**:
- Closed under addition, multiplication, subtraction, division, and all polynomial operations
- Every polynomial equation $a_n x^n + a_{n-1} x^{n-1} + \dots + a_0 = 0$ with complex coefficients has $n$ complex roots (Fundamental Theorem of Algebra)

## Formula(s)

- **Complex number in rectangular form**: $z = a + bi$
- **Complex conjugate**: $\overline{z} = a - bi$
- **Modulus (magnitude)**: $|z| = \sqrt{a^2 + b^2}$
- **Euler's formula**: $e^{i\theta} = \cos\theta + i\sin\theta$
- **Complex number in polar form**: $z = |z|(\cos\theta + i\sin\theta) = |z|e^{i\theta}$
- **De Moivre's Theorem**: $(\cos\theta + i\sin\theta)^n = \cos(n\theta) + i\sin(n\theta)$

## Properties

- **Inclusion hierarchy**: $\mathbb{N} \subset \mathbb{W} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$
- **Closure**: Each system is closed under certain operations. The chain of extensions systematically closes more operations.
- **Countability**: $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$ are **countable** (can be paired with $\mathbb{N}$). $\mathbb{R}$ and $\mathbb{C}$ are **uncountable** (cannot be listed).
- **Algebraic completeness**: $\mathbb{C}$ is algebraically closed — every polynomial equation has a solution in $\mathbb{C}$. $\mathbb{R}$ is not algebraically closed ($x^2 + 1 = 0$ has no real solution).
- **Order**: $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$, $\mathbb{R}$ are **ordered** (given any two distinct numbers, one is greater). $\mathbb{C}$ is **not ordered** — there is no sensible way to say $3 + 4i > 1 + 2i$.
- **Density**: $\mathbb{Q}$ and $\mathbb{R}$ are dense (between any two, there is another). $\mathbb{N}$ and $\mathbb{Z}$ are discrete (they have gaps).
- **Completeness**: $\mathbb{R}$ is **complete** (every Cauchy sequence converges). $\mathbb{Q}$ is not complete (sequences can converge to irrational limits).
- **Cardinality**: $|\mathbb{N}| = |\mathbb{Z}| = |\mathbb{Q}| = \aleph_0$ (countable infinity). $|\mathbb{R}| = |\mathbb{C}| = 2^{\aleph_0}$ (continuum).

## Step-by-Step Worked Examples

### Example 1: Classify Numbers

**Problem**: Classify each number into the smallest number system that contains it: $-3$, $\frac{2}{3}$, $\sqrt{2}$, $0$, $4$, $2 + 3i$, $\pi$.

**Step 1**: $-3$ is an integer but not a natural number. Smallest system: $\mathbb{Z}$.

**Step 2**: $\frac{2}{3}$ is a ratio of integers. Smallest system: $\mathbb{Q}$.

**Step 3**: $\sqrt{2}$ cannot be written as $\frac{a}{b}$. It is irrational. Smallest system: $\mathbb{R}$.

**Step 4**: $0$ is a whole number. Smallest system: $\mathbb{W}$ (or $\mathbb{Z}$ if $\mathbb{W}$ is not considered separate).

**Step 5**: $4$ is a natural number. Smallest system: $\mathbb{N}$.

**Step 6**: $2 + 3i$ has an imaginary part. Smallest system: $\mathbb{C}$.

**Step 7**: $\pi$ is irrational. Smallest system: $\mathbb{R}$.

**Answer**: $-3 \in \mathbb{Z}$, $\frac{2}{3} \in \mathbb{Q}$, $\sqrt{2} \in \mathbb{R}$, $0 \in \mathbb{W}$, $4 \in \mathbb{N}$, $2+3i \in \mathbb{C}$, $\pi \in \mathbb{R}$.

### Example 2: Complex Number Arithmetic

**Problem**: Compute $(3 + 2i) + (1 - 4i)$ and $(3 + 2i)(1 - 4i)$.

**Step 1 (Addition)**: Add real parts and imaginary parts separately.
$$(3 + 2i) + (1 - 4i) = (3 + 1) + (2 - 4)i = 4 - 2i$$

**Step 2 (Multiplication)**: Use FOIL (First, Outer, Inner, Last).
$$(3 + 2i)(1 - 4i) = 3(1) + 3(-4i) + 2i(1) + 2i(-4i)$$

**Step 3**: Simplify each term.
$$= 3 - 12i + 2i - 8i^2$$

**Step 4**: Substitute $i^2 = -1$.
$$= 3 - 10i - 8(-1) = 3 - 10i + 8 = 11 - 10i$$

**Answer**: Sum = $4 - 2i$, Product = $11 - 10i$.

### Example 3: Prove $\sqrt{2}$ is Irrational

**Problem**: Prove that $\sqrt{2}$ cannot be expressed as $\frac{a}{b}$ with integers $a, b$.

**Step 1**: Assume the contrary — that $\sqrt{2} = \frac{a}{b}$ where $a$ and $b$ are integers with no common factors (fraction in lowest terms).

**Step 2**: Square both sides.
$$2 = \frac{a^2}{b^2}$$

**Step 3**: Multiply both sides by $b^2$.
$$2b^2 = a^2$$

**Step 4**: This means $a^2$ is even (divisible by 2). If $a^2$ is even, then $a$ must be even (since odd squared is odd). So $a = 2k$ for some integer $k$.

**Step 5**: Substitute $a = 2k$ into $2b^2 = a^2$.
$$2b^2 = (2k)^2 = 4k^2$$

**Step 6**: Divide both sides by 2.
$$b^2 = 2k^2$$

**Step 7**: This means $b^2$ is even, so $b$ is even.

**Step 8**: Contradiction: both $a$ and $b$ are even, so they share a factor of 2. But we assumed $\frac{a}{b}$ was in lowest terms.

Therefore, our assumption was false — $\sqrt{2}$ cannot be expressed as a rational number.

### Example 4: Complex Conjugate and Modulus

**Problem**: Find the conjugate and modulus of $z = 5 - 12i$.

**Step 1**: The conjugate flips the sign of the imaginary part.
$$\overline{z} = 5 + 12i$$

**Step 2**: The modulus is the distance from the origin in the complex plane.
$$|z| = \sqrt{5^2 + (-12)^2} = \sqrt{25 + 144} = \sqrt{169} = 13$$

**Answer**: $\overline{z} = 5 + 12i$, $|z| = 13$.

### Example 5: Euler's Formula

**Problem**: Express $z = 1 + i$ in polar form using Euler's formula.

**Step 1**: Find the modulus.
$$|z| = \sqrt{1^2 + 1^2} = \sqrt{2}$$

**Step 2**: Find the argument (angle).
$$\theta = \arctan\left(\frac{1}{1}\right) = \frac{\pi}{4}$$

**Step 3**: Apply Euler's formula: $z = |z|e^{i\theta}$.
$$z = \sqrt{2}\, e^{i\pi/4}$$

**Answer**: $1 + i = \sqrt{2}\, e^{i\pi/4}$.

## Visual Interpretation

- **Number line**: $\mathbb{R}$ corresponds to all points on an infinite continuous line. $\mathbb{N}$ are evenly spaced points at integer positions. $\mathbb{Z}$ extends this to the left (negatives). $\mathbb{Q}$ fills some but not all gaps between integers.
- **Complex plane**: $\mathbb{C}$ corresponds to all points on a 2D plane. The $x$-axis is the real part, the $y$-axis is the imaginary part. Real numbers $\mathbb{R}$ lie along the horizontal axis. Purely imaginary numbers lie along the vertical axis.
- **Venn diagram**: Visualize $\mathbb{N}$ as a small circle inside $\mathbb{Z}$ (which adds negatives and zero), inside $\mathbb{Q}$ (which adds fractions), inside $\mathbb{R}$ (which adds irrationals), inside $\mathbb{C}$ (which adds imaginary numbers).

## Common Mistakes

1. **Confusing $\mathbb{N}$ (including or excluding 0)**: Always check which convention a text uses for natural numbers.
2. **Assuming all decimals are rational**: $0.333...$ (repeating) is rational $= 1/3$. But $0.1010010001...$ (non-repeating pattern) is irrational. The key is whether the decimal terminates or repeats.
3. **Thinking $\mathbb{C}$ is "not real"**: Complex numbers are as mathematically valid as reals. They are essential in physics, engineering, and signal processing.
4. **Believing $\mathbb{R}$ is countable**: Cantor's diagonal argument proves there are more real numbers than natural numbers — there are different sizes of infinity.
5. **Forgetting closure**: $\mathbb{Z}$ is not closed under division ($2 \div 3 \notin \mathbb{Z}$). $\mathbb{Q}$ is closed under division but not under taking limits (a limit of rationals can be irrational).
6. **Misapplying order to complex numbers**: You cannot say $3 + 4i > 1 + 2i$ — complex numbers have no natural ordering.
7. **Thinking $\pi$ is exactly $22/7$**: $22/7 \approx 3.142857$ while $\pi \approx 3.141593$. They are close but not equal. $\pi$ is irrational; $22/7$ is rational.

## Interview Questions

### Beginner

**Q1**: What is the difference between $\mathbb{N}$ and $\mathbb{Z}$?
**A**: $\mathbb{N} = \{1, 2, 3, \dots\}$ consists of positive counting numbers. $\mathbb{Z}$ adds zero and negative numbers: $\{\dots, -2, -1, 0, 1, 2, \dots\}$.

**Q2**: Why can't we divide by zero in any number system?
**A**: Division by zero is undefined because if $a/0 = b$, then $b \times 0 = a$, which forces $a = 0$ and $b$ to be anything — a contradiction for nonzero $a$ or non-unique for $a = 0$.

**Q3**: Is $-3$ a natural number? If not, why?
**A**: No. Natural numbers are typically defined as positive integers $\{1, 2, 3, \dots\}$. Negative numbers belong to $\mathbb{Z}$ (integers).

**Q4**: What makes a number rational?
**A**: A number is rational if it can be expressed as a fraction $\frac{a}{b}$ where $a$ and $b$ are integers and $b \neq 0$.

**Q5**: What is $i$ in complex numbers?
**A**: $i$ is the imaginary unit, defined as $i = \sqrt{-1}$. It satisfies $i^2 = -1$.

### Intermediate

**Q1**: Prove that the sum of a rational and an irrational number is irrational.
**A**: Let $r \in \mathbb{Q}$ and $x \notin \mathbb{Q}$. Assume $r + x \in \mathbb{Q}$. Then $x = (r + x) - r$ is the difference of two rationals, hence rational. Contradiction. Therefore $r + x$ is irrational.

**Q2**: Why is $\mathbb{Q}$ countable but $\mathbb{R}$ is not?
**A**: $\mathbb{Q}$ can be arranged in a sequence by listing fractions in a diagonal grid (Cantor's pairing function). $\mathbb{R}$ cannot be listed — Cantor's diagonal argument shows any list of reals misses at least one real number.

**Q3**: Explain the Fundamental Theorem of Algebra.
**A**: Every non-constant polynomial with complex coefficients has at least one complex root. A consequence: a degree-$n$ polynomial has exactly $n$ complex roots (counting multiplicities). This is why $\mathbb{C}$ is "algebraically complete."

**Q4**: What is the conjugate of a complex number, and why is it useful?
**A**: The conjugate of $a + bi$ is $a - bi$. It is used to compute the modulus ($|z|^2 = z\overline{z}$), to rationalize denominators involving complex numbers, and to find real-valued expressions from complex ones.

**Q5**: How do you add and multiply complex numbers?
**A**: Addition: add real and imaginary parts separately. Multiplication: use FOIL and substitute $i^2 = -1$. $(a+bi)+(c+di) = (a+c)+(b+d)i$, $(a+bi)(c+di) = (ac-bd)+(ad+bc)i$.

### Advanced

**Q1**: Explain Cantor's diagonal argument for the uncountability of $\mathbb{R}$.
**A**: Assume $\mathbb{R}$ is countable and list all real numbers between 0 and 1. Construct a new number whose $n$-th decimal digit differs from the $n$-th digit of the $n$-th number in the list. This new number is not in the list, contradicting the assumption that the list contained all reals. Hence $\mathbb{R}$ is uncountable.

**Q2**: What is the significance of $\mathbb{C}$ being algebraically closed while $\mathbb{R}$ is not?
**A**: Algebraic closure means every polynomial equation has a solution within the system. In $\mathbb{R}$, $x^2 + 1 = 0$ has no solution. In $\mathbb{C}$, it has solutions $x = i$ and $x = -i$. This completeness makes $\mathbb{C}$ the natural setting for much of mathematics.

**Q3**: How does the concept of number systems relate to floating-point arithmetic in computing?
**A**: Floating-point numbers (IEEE 754) approximate $\mathbb{R}$ using a finite subset of $\mathbb{Q}$. This creates issues: rounding errors, catastrophic cancellation, and the "floating-point paradox" where $(a + b) + c \neq a + (b + c)$ due to precision limits. Understanding the gap between $\mathbb{R}$ and floating-point representation is critical for numerical stability in ML.

## Practice Problems

### Easy - 5 Questions

**E1**: Classify $-7$ into the smallest number system.
**E2**: Is $0.5$ a rational number? Explain.
**E3**: Simplify $(2 + i) + (3 - 2i)$.
**E4**: What is the conjugate of $4 + 7i$?
**E5**: Is $\frac{22}{7}$ equal to $\pi$? Why or why not?

### Medium - 5 Questions

**M1**: Compute $(3 - 2i)(4 + i)$.
**M2**: Find $|3 - 4i|$.
**M3**: Prove that $\sqrt{3}$ is irrational.
**M4**: Express $\frac{1}{2 - i}$ in the form $a + bi$.
**M5**: Show that the set of integers $\mathbb{Z}$ is countably infinite by describing a bijection with $\mathbb{N}$.

### Hard - 3 Questions

**H1**: Prove that $\log_2 3$ is irrational.
**H2**: Show that $\mathbb{Q}$ is dense in $\mathbb{R}$ — i.e., between any two real numbers there exists a rational number.
**H3**: Express $\left(\frac{1}{2} + i\frac{\sqrt{3}}{2}\right)^{2024}$ in simplest form.

## Solutions

### Easy Solutions

**E1**: $-7$ is an integer. Smallest system: $\mathbb{Z}$.

**E2**: Yes. $0.5 = \frac{1}{2}$, which is a ratio of two integers with nonzero denominator. All terminating decimals are rational.

**E3**: $(2 + i) + (3 - 2i) = (2 + 3) + (1 - 2)i = 5 - i$.

**E4**: The conjugate of $4 + 7i$ is $4 - 7i$.

**E5**: No. $\frac{22}{7} \approx 3.142857$ is a rational approximation of $\pi \approx 3.141593$, but $\pi$ is irrational (non-terminating, non-repeating decimal), so they cannot be equal.

### Medium Solutions

**M1**: $(3 - 2i)(4 + i) = 12 + 3i - 8i - 2i^2 = 12 - 5i - 2(-1) = 12 - 5i + 2 = 14 - 5i$.

**M2**: $|3 - 4i| = \sqrt{3^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.

**M3**: Assume $\sqrt{3} = a/b$ in lowest terms. Then $3b^2 = a^2$. So $a^2$ is divisible by 3, hence $a$ is divisible by 3: $a = 3k$. Then $3b^2 = 9k^2 \Rightarrow b^2 = 3k^2$, so $b$ is also divisible by 3. Contradiction — $a$ and $b$ share factor 3. Hence $\sqrt{3}$ is irrational.

**M4**: Multiply numerator and denominator by the conjugate:
$$\frac{1}{2 - i} \cdot \frac{2 + i}{2 + i} = \frac{2 + i}{(2 - i)(2 + i)} = \frac{2 + i}{4 - i^2} = \frac{2 + i}{4 + 1} = \frac{2 + i}{5} = \frac{2}{5} + \frac{1}{5}i$$

**M5**: Define $f: \mathbb{N} \to \mathbb{Z}$ as:
$$f(n) = \begin{cases}
\frac{n}{2} & \text{if } n \text{ is even} \\
-\frac{n-1}{2} & \text{if } n \text{ is odd}
\end{cases}$$
This maps $\mathbb{N} = \{1, 2, 3, 4, 5, \dots\}$ to $\mathbb{Z} = \{0, 1, -1, 2, -2, \dots\}$, a bijection proving countability.

### Hard Solutions

**H1**: Assume $\log_2 3 = \frac{a}{b}$ with $a, b \in \mathbb{Z}^+$ in lowest terms. Then $2^{a/b} = 3 \Rightarrow 2^a = 3^b$. The left side is even (power of 2), the right side is odd (power of 3). Contradiction. Hence $\log_2 3$ is irrational.

**H2**: Let $x < y$ be real numbers. Choose an integer $n$ such that $n > \frac{1}{y - x}$. Then $\frac{1}{n} < y - x$. Let $m = \lceil nx \rceil$ (the smallest integer $\geq nx$). Then $nx \leq m < nx + 1 \leq nx + n(y-x) = ny$. So $x \leq \frac{m}{n} \leq y$. Since $x < y$, we actually have $x < \frac{m}{n} < y$. Thus $\frac{m}{n} \in \mathbb{Q}$ lies strictly between $x$ and $y$.

**H3**: Let $z = \frac{1}{2} + i\frac{\sqrt{3}}{2}$. Write in polar form: $|z| = \sqrt{(1/2)^2 + (\sqrt{3}/2)^2} = \sqrt{1/4 + 3/4} = 1$. $\theta = \arctan\left(\frac{\sqrt{3}/2}{1/2}\right) = \arctan(\sqrt{3}) = \pi/3$. So $z = e^{i\pi/3}$.
Then $z^{2024} = e^{i(2024\pi/3)} = e^{i(674\pi + 2\pi/3)} = e^{i(2\pi/3)} \cdot e^{i(674\pi)} = e^{i(2\pi/3)} \cdot 1$ (since $674\pi$ is even multiple of $\pi$).
$e^{i(2\pi/3)} = \cos(2\pi/3) + i\sin(2\pi/3) = -\frac{1}{2} + i\frac{\sqrt{3}}{2}$.

## Related Concepts

- **Set Theory**: The language of sets provides the formal framework for defining number systems.
- **Algebraic Structures**: Groups, rings, and fields generalize the properties of number systems.
- **Number Line**: The geometric representation of real numbers.
- **Exponents and Logarithms**: Operations defined on real numbers.
- **Polynomial Equations**: Solving these drove the extension from $\mathbb{R}$ to $\mathbb{C}$.

## Next Concepts

- **Real Numbers** (MATH-008): Detailed study of $\mathbb{R}$, including completeness and properties.
- **Complex Analysis**: Calculus with complex numbers.
- **Abstract Algebra**: Group theory, ring theory, field theory.
- **Numerical Analysis**: Floating-point arithmetic and error analysis.
- **p-adic Numbers**: An alternative completion of $\mathbb{Q}$.

## Summary

Number systems form a hierarchy: $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$. Each extension was motivated by solving a limitation of the previous system:

| System | Elements | Closed under | Limitation addressed |
|--------|----------|-------------|---------------------|
| $\mathbb{N}$ | $1,2,3,\dots$ | $+, \times$ | Basic counting |
| $\mathbb{Z}$ | $\dots,-2,-1,0,1,2,\dots$ | $+,-,\times$ | Subtraction |
| $\mathbb{Q}$ | $\frac{a}{b}$ | $+,-,\times,\div$ | Division |
| $\mathbb{R}$ | All decimals | $+,-,\times,\div$, limits | Irrational numbers |
| $\mathbb{C}$ | $a+bi$ | All algebraic operations | $x^2 = -1$ |

The key distinction is between countable sets ($\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$) and uncountable sets ($\mathbb{R}$, $\mathbb{C}$). Complex numbers are algebraically complete, making them the natural domain for polynomial equations.

## Key Takeaways

- Number systems are hierarchically organized, with each extending the previous to enable new operations.
- $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$.
- $\mathbb{N}$, $\mathbb{Z}$, and $\mathbb{Q}$ are countably infinite; $\mathbb{R}$ and $\mathbb{C}$ are uncountably infinite.
- A number is rational if its decimal expansion terminates or repeats; otherwise it is irrational.
- $\mathbb{R}$ is complete (no gaps) and ordered. $\mathbb{C}$ is complete but not ordered.
- $\mathbb{C}$ is algebraically closed: every polynomial equation has a complex solution.
- In ML, most data features are real-valued ($\mathbb{R}$), weights use floating-point approximations of $\mathbb{R}$, and understanding numerical precision requires knowing the limits of finite rational representations.
- Complex numbers are increasingly used in ML for signal processing, frequency-domain analysis, and complex-valued neural networks.
