# Concept: Complex Numbers

## Concept ID

MATH-009

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Foundations

## Learning Objectives

- Understand the definition of complex numbers and the imaginary unit $i$
- Represent complex numbers in $a+bi$ form and on the Argand diagram
- Perform arithmetic operations (addition, subtraction, multiplication, division) with complex numbers
- Compute the modulus and conjugate of a complex number
- Express complex numbers using Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$
- Apply complex numbers to real-world problems and AI/ML contexts

## Prerequisites

- Basic algebra: understanding variables, equations, and the distributive property
- Trigonometric functions: $\sin\theta$, $\cos\theta$, and the unit circle
- Real numbers and the number line
- Basic set notation and the concept of number systems ($\mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{R}$)

## Definition

A **complex number** is a number of the form $a + bi$, where $a$ and $b$ are real numbers and $i$ is the **imaginary unit** satisfying $i^2 = -1$.

- $a$ is called the **real part**, denoted $\text{Re}(z)$
- $b$ is called the **imaginary part**, denoted $\text{Im}(z)$
- $i$ (the imaginary unit) is defined as $i = \sqrt{-1}$

The set of all complex numbers is denoted by $\mathbb{C}$. Formally:

$$
\mathbb{C} = \{a + bi \mid a, b \in \mathbb{R},\; i^2 = -1\}
$$

Any real number is also a complex number (with $b = 0$), so $\mathbb{R} \subset \mathbb{C}$.

## Intuition

Real numbers live on a one-dimensional number line. Complex numbers extend this to a **two-dimensional plane**. Think of the real part $a$ as the east-west coordinate and the imaginary part $b$ as the north-south coordinate. This plane is called the **complex plane** (or Argand diagram).

Why invent $i$? Because some equations like $x^2 + 1 = 0$ have no real solutions (no real number squared gives $-1$). By introducing $i$, every polynomial equation gains a solution — this is the **Fundamental Theorem of Algebra**.

Just as negative numbers let us go left of zero on the number line, complex numbers let us go "off the line" entirely, rotating into a second dimension.

## Why This Concept Matters

Complex numbers are not just a mathematical curiosity — they are a cornerstone of modern science and engineering:

- **Electrical engineering**: AC circuit analysis uses complex impedance
- **Signal processing**: Fourier transforms rely on complex exponentials
- **Quantum mechanics**: The wavefunction is inherently complex-valued
- **Control theory**: System stability is analyzed in the complex $s$-plane
- **Machine learning**: Complex-valued neural networks and quantum ML models use complex numbers

Many problems that seem difficult in the real domain become elegantly simple in the complex domain.

## Historical Background

The story of complex numbers spans centuries:

- **1st century CE**: Greek mathematician Heron of Alexandria incidentally encountered $\sqrt{81 - 144}$ but treated it as $144 - 81$.
- **1545**: Gerolamo Cardano attempted to solve cubic equations and encountered square roots of negative numbers in his *Ars Magna*.
- **1572**: Rafael Bombelli formalized rules for manipulating "imaginary" expressions.
- **1637**: René Descartes coined the term "imaginary" as a derogatory label.
- **1748**: Leonhard Euler published Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$.
- **1831**: Carl Friedrich Gauss introduced the term "complex number" and the Argand diagram representation.
- **1900s**: Complex analysis became fully rigorous and essential to physics and engineering.

The name "imaginary" stuck, but these numbers are no less real (in the physical sense) than negative numbers or irrationals.

## Real World Examples

### 1. Electrical Impedance

In AC circuits, resistors have real impedance $R$, but capacitors and inductors have imaginary impedance:

$$
Z = R + iX
$$

where $X$ is the reactance. The magnitude $|Z| = \sqrt{R^2 + X^2}$ gives the total opposition to current flow.

### 2. Rotations in 2D

Multiplying by $i$ rotates a point $90^\circ$ counterclockwise in the complex plane:

$$
i \cdot (a + bi) = -b + ai
$$

This is used in computer graphics for 2D rotations.

### 3. Fractals (Mandelbrot Set)

The Mandelbrot set is defined by iterating the complex function $z_{n+1} = z_n^2 + c$, where all variables are complex. The boundary of this set exhibits infinite complexity and self-similarity.

## AI/ML Relevance

Complex numbers appear in several cutting-edge ML domains:

1.  **Complex-Valued Neural Networks (CVNNs)**: Some data (e.g., audio spectrograms, MRI signals, radar data) is naturally complex. CVNNs process this directly rather than splitting into real and imaginary parts, often achieving better representational capacity.

2.  **Quantum Machine Learning**: Quantum states are described by complex amplitudes. Qubit states are unit vectors in $\mathbb{C}^2$:
    $$
    |\psi\rangle = \alpha|0\rangle + \beta|1\rangle,\quad \alpha,\beta \in \mathbb{C}
    $$
    Quantum ML models leverage superposition and entanglement, which require complex arithmetic.

3.  **Signal Processing for ML Features**: Fourier and wavelet transforms (used to preprocess audio, images, and time series data) operate in the complex domain. The magnitude and phase components carry different information.

4.  **Adversarial Robustness**: Complex-valued representations have shown promise in improving robustness against adversarial attacks in certain architectures.

## Mathematical Explanation

### The Imaginary Unit

The imaginary unit $i$ is defined by the property:

$$
i^2 = -1
$$

From this, we derive the cyclic pattern of powers of $i$:

$$
i^1 = i,\quad i^2 = -1,\quad i^3 = -i,\quad i^4 = 1,\quad i^5 = i, \ldots
$$

The cycle repeats every 4 powers.

### Standard Form

Every complex number can be written as $z = a + bi$ where $a,b \in \mathbb{R}$.

- Equality: $a + bi = c + di$ if and only if $a = c$ and $b = d$
- Zero: $0 + 0i$ is the additive identity

### Argand Diagram (Complex Plane)

The complex plane has:
- **Horizontal axis**: Real part ($\text{Re}$)
- **Vertical axis**: Imaginary part ($\text{Im}$)

The complex number $z = a + bi$ corresponds to the point $(a, b)$.

### Modulus (Absolute Value)

The modulus of $z = a + bi$ is its distance from the origin:

$$
|z| = \sqrt{a^2 + b^2}
$$

Properties:
- $|z| \geq 0$, with $|z| = 0$ iff $z = 0$
- $|z_1 z_2| = |z_1| \cdot |z_2|$
- $|z_1 + z_2| \leq |z_1| + |z_2|$ (triangle inequality)

### Complex Conjugate

The conjugate of $z = a + bi$ is denoted $\bar{z}$ and defined as:

$$
\bar{z} = a - bi
$$

Geometrically, conjugation reflects across the real axis.

Properties:
- $\overline{z_1 + z_2} = \bar{z}_1 + \bar{z}_2$
- $\overline{z_1 \cdot z_2} = \bar{z}_1 \cdot \bar{z}_2$
- $z \cdot \bar{z} = |z|^2 = a^2 + b^2$ (a real number)
- $\overline{\bar{z}} = z$

### Arithmetic Operations

Let $z_1 = a + bi$ and $z_2 = c + di$.

**Addition**:
$$
z_1 + z_2 = (a + c) + (b + d)i
$$
(Vector addition in the plane.)

**Subtraction**:
$$
z_1 - z_2 = (a - c) + (b - d)i
$$

**Multiplication**:
$$
\begin{aligned}
z_1 \cdot z_2 &= (a + bi)(c + di) \\
&= ac + adi + bci + bdi^2 \\
&= ac + adi + bci - bd \\
&= (ac - bd) + (ad + bc)i
\end{aligned}
$$

**Division**:
To divide $z_1 / z_2$, multiply numerator and denominator by the conjugate of the denominator:
$$
\frac{z_1}{z_2} = \frac{a + bi}{c + di} \cdot \frac{c - di}{c - di} = \frac{(a + bi)(c - di)}{c^2 + d^2} = \frac{ac + bd}{c^2 + d^2} + \frac{bc - ad}{c^2 + d^2}i
$$

### Euler's Formula

One of the most beautiful results in mathematics:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

This connects exponential functions to trigonometric functions.

**Polar Form**: Using Euler's formula, any complex number can be written as:

$$
z = r(\cos\theta + i\sin\theta) = r e^{i\theta}
$$

where $r = |z|$ is the modulus and $\theta = \arg(z)$ is the argument (angle from the positive real axis).

**De Moivre's Theorem**: For integer $n$:

$$
(r e^{i\theta})^n = r^n e^{in\theta} = r^n(\cos n\theta + i\sin n\theta)
$$

## Formula(s)

| Concept | Formula |
|---|---|
| Standard form | $z = a + bi$ |
| Modulus | $|z| = \sqrt{a^2 + b^2}$ |
| Conjugate | $\bar{z} = a - bi$ |
| Addition | $(a+bi)+(c+di) = (a+c)+(b+d)i$ |
| Multiplication | $(a+bi)(c+di) = (ac-bd)+(ad+bc)i$ |
| Division | $\frac{a+bi}{c+di} = \frac{ac+bd}{c^2+d^2} + \frac{bc-ad}{c^2+d^2}i$ |
| Euler's formula | $e^{i\theta} = \cos\theta + i\sin\theta$ |
| Polar form | $z = r e^{i\theta}$ |
| De Moivre | $(r e^{i\theta})^n = r^n e^{in\theta}$ |

## Properties

1. **Closure**: The sum or product of two complex numbers is complex.
2. **Commutativity**: $z_1 + z_2 = z_2 + z_1$ and $z_1 z_2 = z_2 z_1$
3. **Associativity**: $(z_1 + z_2) + z_3 = z_1 + (z_2 + z_3)$ and $(z_1 z_2) z_3 = z_1 (z_2 z_3)$
4. **Distributivity**: $z_1(z_2 + z_3) = z_1 z_2 + z_1 z_3$
5. **Additive identity**: $0 + 0i$
6. **Multiplicative identity**: $1 + 0i$
7. **Every non-zero complex number has a multiplicative inverse**: $z^{-1} = \frac{\bar{z}}{|z|^2}$
8. **Conjugate symmetry**: $\overline{\overline{z}} = z$
9. **Real numbers**: If $b = 0$, $z$ is real; if $a = 0$, $z$ is purely imaginary

## Step-by-Step Worked Examples

### Example 1: Arithmetic Operations

Given $z_1 = 3 + 4i$ and $z_2 = 1 - 2i$, find $z_1 + z_2$, $z_1 - z_2$, $z_1 \cdot z_2$, and $z_1 / z_2$.

**Step 1: Addition**
$$
z_1 + z_2 = (3 + 4i) + (1 - 2i) = (3 + 1) + (4 - 2)i = 4 + 2i
$$

**Step 2: Subtraction**
$$
z_1 - z_2 = (3 + 4i) - (1 - 2i) = (3 - 1) + (4 - (-2))i = 2 + 6i
$$

**Step 3: Multiplication**
$$
\begin{aligned}
z_1 \cdot z_2 &= (3 + 4i)(1 - 2i) \\
&= 3(1) + 3(-2i) + 4i(1) + 4i(-2i) \\
&= 3 - 6i + 4i - 8i^2 \\
&= 3 - 2i - 8(-1) \\
&= 3 - 2i + 8 \\
&= 11 - 2i
\end{aligned}
$$

**Step 4: Division**
$$
\begin{aligned}
\frac{z_1}{z_2} &= \frac{3 + 4i}{1 - 2i} \cdot \frac{1 + 2i}{1 + 2i} \\
&= \frac{(3 + 4i)(1 + 2i)}{(1)^2 + (2)^2} \\
&= \frac{3 + 6i + 4i + 8i^2}{1 + 4} \\
&= \frac{3 + 10i - 8}{5} \\
&= \frac{-5 + 10i}{5} \\
&= -1 + 2i
\end{aligned}
$$

**Check**: Multiply $(-1 + 2i)(1 - 2i) = -1 + 2i + 2i - 4i^2 = -1 + 4i + 4 = 3 + 4i$ ✓

---

### Example 2: Modulus and Conjugate

Given $z = 5 - 12i$, find $|z|$, $\bar{z}$, and verify $z \cdot \bar{z} = |z|^2$.

**Step 1: Modulus**
$$
|z| = \sqrt{5^2 + (-12)^2} = \sqrt{25 + 144} = \sqrt{169} = 13
$$

**Step 2: Conjugate**
$$
\bar{z} = 5 + 12i
$$

**Step 3: Verify $z \cdot \bar{z} = |z|^2$**
$$
z \cdot \bar{z} = (5 - 12i)(5 + 12i) = 25 + 60i - 60i - 144i^2 = 25 + 144 = 169 = 13^2 = |z|^2
$$

---

### Example 3: Powers of $i$ and Euler's Formula

**(a)** Simplify $i^{37}$.

**Step 1**: Find the remainder when 37 is divided by 4:
$$
37 \div 4 = 9 \text{ remainder } 1
$$

**Step 2**: Since the cycle of $i^n$ repeats every 4:
$$
i^{37} = i^{4\cdot 9 + 1} = (i^4)^9 \cdot i^1 = 1^9 \cdot i = i
$$

**(b)** Express $z = -1 + i\sqrt{3}$ in polar form using Euler's formula.

**Step 1: Find modulus $r$**
$$
r = \sqrt{(-1)^2 + (\sqrt{3})^2} = \sqrt{1 + 3} = \sqrt{4} = 2
$$

**Step 2: Find argument $\theta$**
$$
\cos\theta = \frac{a}{r} = \frac{-1}{2} = -\frac{1}{2},\quad \sin\theta = \frac{b}{r} = \frac{\sqrt{3}}{2}
$$

The angle with $\cos\theta = -\frac{1}{2}$ and $\sin\theta = \frac{\sqrt{3}}{2}$ is $\theta = \frac{2\pi}{3}$ (120° in the second quadrant).

**Step 3: Polar form**
$$
z = 2\left(\cos\frac{2\pi}{3} + i\sin\frac{2\pi}{3}\right) = 2e^{i\frac{2\pi}{3}}
$$

**(c)** Use De Moivre's theorem to find $z^3$.

$$
z^3 = \left(2e^{i\frac{2\pi}{3}}\right)^3 = 2^3 e^{i\cdot 3 \cdot \frac{2\pi}{3}} = 8 e^{i2\pi} = 8(\cos 2\pi + i\sin 2\pi) = 8(1 + 0) = 8
$$

This makes sense because $(-1 + i\sqrt{3})^3 = 8$ is a real number.

---

### Example 4: Solving Quadratic Equations

Solve $x^2 + 2x + 5 = 0$ over the complex numbers.

**Step 1: Apply the quadratic formula**
$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} = \frac{-2 \pm \sqrt{4 - 20}}{2} = \frac{-2 \pm \sqrt{-16}}{2}
$$

**Step 2: Simplify the square root of the negative number**
$$
\sqrt{-16} = \sqrt{16} \cdot \sqrt{-1} = 4i
$$

**Step 3: Write the solutions**
$$
x = \frac{-2 \pm 4i}{2} = -1 \pm 2i
$$

So the solutions are $x = -1 + 2i$ and $x = -1 - 2i$ (a conjugate pair).

**Step 4: Verify**
For $x = -1 + 2i$:
$$
(-1 + 2i)^2 + 2(-1 + 2i) + 5 = (1 - 4i - 4) + (-2 + 4i) + 5 = ( -3 - 4i) + (-2 + 4i) + 5 = 0
$$

## Visual Interpretation

### The Argand Diagram

The complex number $z = a + bi$ is plotted as the point $(a, b)$ in a 2D coordinate system:

```
      Im
      ^
      |
   b  |--------• (a, b)
      |       /|
      |      / |
      |  r  /  |
      |    /   |
      |   /θ   |
      |  /-----+------>
      |             Re
                a
```

- The **modulus** $r = |z|$ is the distance from the origin to the point
- The **argument** $\theta = \arg(z)$ is the angle measured from the positive real axis
- $a = r\cos\theta$ and $b = r\sin\theta$
- The **conjugate** $\bar{z} = a - bi$ is the reflection of $z$ across the real axis

### Geometric Interpretation of Operations

- **Addition**: Vector addition — place the second vector at the tip of the first
- **Multiplication by $i$**: Rotates the point $90^\circ$ counterclockwise
- **Multiplication in general**: Multiply moduli and add arguments:
  $$
  |z_1 z_2| = |z_1| \cdot |z_2|,\quad \arg(z_1 z_2) = \arg(z_1) + \arg(z_2)
  $$
- **Division**: Divide moduli and subtract arguments

## Common Mistakes

1. **Incorrectly simplifying $i^2$**: Remember $i^2 = -1$, not $1$. A common error is treating $i$ like a regular variable and writing $i^2 = i^2$ without simplifying.

2. **Forgetting to multiply numerator and denominator by the conjugate**: When dividing complex numbers, you must multiply both numerator and denominator by the conjugate of the denominator, not just the denominator alone.

3. **Confusing the conjugate with the negative**: The conjugate of $a + bi$ is $a - bi$, not $-a - bi$ or $-a + bi$.

4. **Mistaking the modulus**: The modulus is $\sqrt{a^2 + b^2}$, not $\sqrt{a^2 + (bi)^2} = \sqrt{a^2 - b^2}$. The imaginary part $b$ is a real number — do not include $i$ inside the square root.

5. **Errors in equating real and imaginary parts**: For $a + bi = c + di$, we need $a = c$ AND $b = d$. Both conditions must hold. Do not mix real and imaginary parts.

6. **Incorrectly applying the quadratic formula**: When the discriminant is negative, $\sqrt{-D} = i\sqrt{D}$, not $\sqrt{D}i$ (though these are equivalent by commutativity — the real error is forgetting $i$ entirely).

7. **Assuming complex numbers have ordering**: There is no natural order on $\mathbb{C}$. Statements like $3 + 4i < 5 + 2i$ are meaningless. You can only compare moduli.

## Interview Questions

### Beginner

1. **Q**: What is the imaginary unit $i$ and what is its defining property?
   **A**: $i$ is defined as $i = \sqrt{-1}$, with the key property $i^2 = -1$.

2. **Q**: What is the conjugate of $3 - 7i$?
   **A**: $3 + 7i$. The conjugate flips the sign of the imaginary part.

3. **Q**: How do you add two complex numbers $(2 + 3i)$ and $(4 - 5i)$?
   **A**: $(2 + 4) + (3 - 5)i = 6 - 2i$.

4. **Q**: What is the modulus of $3 + 4i$?
   **A**: $|3 + 4i| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$.

5. **Q**: Simplify $i^{23}$.
   **A**: $23 \div 4 = 5$ remainder $3$, so $i^{23} = i^3 = -i$.

### Intermediate

1. **Q**: Express $\frac{2 + i}{1 - i}$ in $a + bi$ form.
   **A**: Multiply by $\frac{1 + i}{1 + i}$: $\frac{(2+i)(1+i)}{1^2+1^2} = \frac{2+2i+i+i^2}{2} = \frac{2+3i-1}{2} = \frac{1+3i}{2} = \frac{1}{2} + \frac{3}{2}i$.

2. **Q**: Find the real numbers $x$ and $y$ such that $(x + yi)(2 - i) = 7 + 4i$.
   **A**: Expand: $2x - xi + 2yi - yi^2 = 2x + y + (-x + 2y)i = 7 + 4i$. Equating: $2x + y = 7$ and $-x + 2y = 4$. Solve: $x = 2$, $y = 3$.

3. **Q**: What is the geometric effect of multiplying a complex number by $i$?
   **A**: Multiplying by $i$ rotates the point $90^\circ$ counterclockwise about the origin in the complex plane.

4. **Q**: Prove that $|z_1 z_2| = |z_1| \cdot |z_2|$ using polar form.
   **A**: Let $z_1 = r_1 e^{i\theta_1}$ and $z_2 = r_2 e^{i\theta_2}$. Then $z_1 z_2 = r_1 r_2 e^{i(\theta_1 + \theta_2)}$, so $|z_1 z_2| = r_1 r_2 = |z_1| \cdot |z_2|$.

5. **Q**: Find the square roots of $i$.
   **A**: Let $\sqrt{i} = a + bi$. Then $(a+bi)^2 = a^2 - b^2 + 2abi = i$. So $a^2 - b^2 = 0$ and $2ab = 1$. Solving: $a = b = \frac{1}{\sqrt{2}}$ or $a = b = -\frac{1}{\sqrt{2}}$. Thus $\sqrt{i} = \pm\frac{1}{\sqrt{2}}(1 + i)$.

### Advanced

1. **Q**: Prove Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ using the Maclaurin series expansions of $e^x$, $\sin x$, and $\cos x$.
   **A**: 
   $$
   e^{i\theta} = \sum_{n=0}^{\infty} \frac{(i\theta)^n}{n!} = 1 + i\theta - \frac{\theta^2}{2!} - i\frac{\theta^3}{3!} + \frac{\theta^4}{4!} + i\frac{\theta^5}{5!} - \cdots
   $$
   Group real and imaginary terms:
   $$
   e^{i\theta} = \left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right) + i\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right) = \cos\theta + i\sin\theta
   $$

2. **Q**: Show that the set $\mathbb{C}$ with addition and multiplication forms a field. Which property fails for $\mathbb{R}^2$ with component-wise multiplication?
   **A**: $\mathbb{C}$ satisfies all field axioms: closure, associativity, commutativity, distributivity, identities (0 and 1), and inverses (additive for all, multiplicative for non-zero). In $\mathbb{R}^2$ with component-wise multiplication, $(1,0)$ is the identity, but $(0,1)$ has no inverse since $(0,1)\cdot(a,b) = (0,0) \neq (1,0)$ for any $(a,b)$.

3. **Q**: Describe how complex numbers arise in quantum ML. Explain why qubit amplitudes must be complex.
   **A**: Quantum states evolve via unitary transformations: $|\psi'\rangle = U|\psi\rangle$. Unitary matrices satisfy $U^\dagger U = I$, where $\dagger$ denotes conjugate transpose. This preserves total probability $|\alpha|^2 + |\beta|^2 = 1$. If amplitudes were real, only orthogonal transformations ($O(2)$) would be possible, but quantum mechanics requires the full unitary group $U(2)$. Complex phases encode interference patterns essential for quantum algorithms. In quantum ML, variational quantum circuits use complex parameters to represent and optimize quantum states for learning tasks.

## Practice Problems

### Easy - 5 Questions

1. Write the real and imaginary parts of $z = -5 + 2i$.

2. Add $(3 - 7i) + (-2 + 4i)$ and express in $a+bi$ form.

3. Find the conjugate of $z = 6 + 8i$ and compute $z\bar{z}$.

4. Compute $|5 - 12i|$.

5. Simplify $i^{50}$.

### Medium - 5 Questions

6. Multiply $(2 - 3i)(4 + 5i)$ and express the result in $a+bi$ form.

7. Divide $\frac{3 + 2i}{1 - i}$ and write in $a+bi$ form.

8. Solve $x^2 + 4x + 13 = 0$ over the complex numbers.

9. Express $z = 1 + i$ in polar form $re^{i\theta}$.

10. If $z = 2 + 3i$ and $w = 1 - i$, find $|z \cdot w|$.

### Hard - 3 Questions

11. Find all cube roots of $z = -8$. Express them in $a+bi$ form.

12. Prove that for any complex number $z$, $\overline{z^n} = (\bar{z})^n$ for positive integer $n$ using induction.

13. Let $f(z) = \frac{z - i}{z + i}$. Find the set of all $z \in \mathbb{C}$ such that $f(z)$ is purely real (i.e., its imaginary part is 0).

## Solutions

### Solutions to Easy Problems

**Solution 1**: 
Real part $\text{Re}(z) = -5$, Imaginary part $\text{Im}(z) = 2$.

**Solution 2**:
$$
(3 - 7i) + (-2 + 4i) = (3 - 2) + (-7 + 4)i = 1 - 3i
$$

**Solution 3**:
Conjugate $\bar{z} = 6 - 8i$.
$$
z\bar{z} = (6 + 8i)(6 - 8i) = 36 - 48i + 48i - 64i^2 = 36 + 64 = 100
$$

**Solution 4**:
$$
|5 - 12i| = \sqrt{5^2 + (-12)^2} = \sqrt{25 + 144} = \sqrt{169} = 13
$$

**Solution 5**:
$50 \div 4 = 12$ remainder $2$, so $i^{50} = i^2 = -1$.

### Solutions to Medium Problems

**Solution 6**:
$$
\begin{aligned}
(2 - 3i)(4 + 5i) &= 2(4) + 2(5i) - 3i(4) - 3i(5i) \\
&= 8 + 10i - 12i - 15i^2 \\
&= 8 - 2i + 15 \\
&= 23 - 2i
\end{aligned}
$$

**Solution 7**:
$$
\begin{aligned}
\frac{3 + 2i}{1 - i} &= \frac{3 + 2i}{1 - i} \cdot \frac{1 + i}{1 + i} \\
&= \frac{(3 + 2i)(1 + i)}{1^2 + 1^2} \\
&= \frac{3 + 3i + 2i + 2i^2}{2} \\
&= \frac{3 + 5i - 2}{2} \\
&= \frac{1 + 5i}{2} = \frac{1}{2} + \frac{5}{2}i
\end{aligned}
$$

**Solution 8**:
$$
x = \frac{-4 \pm \sqrt{16 - 52}}{2} = \frac{-4 \pm \sqrt{-36}}{2} = \frac{-4 \pm 6i}{2} = -2 \pm 3i
$$

Solutions: $x = -2 + 3i$ and $x = -2 - 3i$.

**Solution 9**:
$r = \sqrt{1^2 + 1^2} = \sqrt{2}$, $\theta = \tan^{-1}(1/1) = \pi/4$.
$$
z = \sqrt{2} e^{i\pi/4}
$$

**Solution 10**:
$z \cdot w = (2 + 3i)(1 - i) = 2 - 2i + 3i - 3i^2 = 2 + i + 3 = 5 + i$.
$$
|z \cdot w| = |5 + i| = \sqrt{5^2 + 1^2} = \sqrt{26}
$$

Alternatively: $|z| = \sqrt{13}$, $|w| = \sqrt{2}$, so $|z \cdot w| = \sqrt{13} \cdot \sqrt{2} = \sqrt{26}$.

### Solutions to Hard Problems

**Solution 11**:
$z = -8 = 8(\cos\pi + i\sin\pi) = 8e^{i\pi}$.

The three cube roots are:
$$
w_k = \sqrt[3]{8} e^{i(\pi + 2\pi k)/3} = 2 e^{i(\pi + 2\pi k)/3},\quad k = 0, 1, 2
$$

For $k = 0$:
$$
w_0 = 2e^{i\pi/3} = 2\left(\cos\frac{\pi}{3} + i\sin\frac{\pi}{3}\right) = 2\left(\frac{1}{2} + i\frac{\sqrt{3}}{2}\right) = 1 + i\sqrt{3}
$$

For $k = 1$:
$$
w_1 = 2e^{i\pi} = 2(\cos\pi + i\sin\pi) = 2(-1 + 0) = -2
$$

For $k = 2$:
$$
w_2 = 2e^{i5\pi/3} = 2\left(\cos\frac{5\pi}{3} + i\sin\frac{5\pi}{3}\right) = 2\left(\frac{1}{2} - i\frac{\sqrt{3}}{2}\right) = 1 - i\sqrt{3}
$$

The three cube roots of $-8$ are $1 \pm i\sqrt{3}$ and $-2$.

**Solution 12**:

Base case $n = 1$: $\overline{z^1} = \bar{z} = (\bar{z})^1$. ✓

Inductive step: Assume $\overline{z^n} = (\bar{z})^n$ for some $n \geq 1$.

Then:
$$
\overline{z^{n+1}} = \overline{z^n \cdot z} = \overline{z^n} \cdot \bar{z}
$$

Using the property $\overline{w_1 w_2} = \bar{w}_1 \bar{w}_2$ and the induction hypothesis:
$$
\overline{z^{n+1}} = (\bar{z})^n \cdot \bar{z} = (\bar{z})^{n+1}
$$

Thus the statement holds for $n+1$. By induction, $\overline{z^n} = (\bar{z})^n$ for all positive integers $n$.

**Solution 13**:

$$f(z) = \frac{z - i}{z + i}$$

Substitute $z = x + yi$:

$$
f(z) = \frac{x + yi - i}{x + yi + i} = \frac{x + (y-1)i}{x + (y+1)i}
$$

Multiply numerator and denominator by the conjugate of the denominator:

$$
f(z) = \frac{[x + (y-1)i][x - (y+1)i]}{x^2 + (y+1)^2}
$$

Expand the numerator:

$$
\begin{aligned}
&= \frac{x^2 - x(y+1)i + x(y-1)i - (y-1)(y+1)i^2}{x^2 + (y+1)^2} \\
&= \frac{x^2 + xy i - x i - xy i - x i + (y^2 - 1)}{x^2 + (y+1)^2} \\
&= \frac{x^2 + y^2 - 1 - 2x i}{x^2 + (y+1)^2}
\end{aligned}
$$

For $f(z)$ to be purely real, the imaginary part must be $0$:

$$
\frac{-2x}{x^2 + (y+1)^2} = 0 \implies x = 0
$$

So the set is all points on the imaginary axis ($x = 0$), i.e., $z = yi$ for $y \in \mathbb{R}$, except $z = -i$ (where the denominator is zero).

## Related Concepts

- **Real Numbers ($\mathbb{R}$)**: The subset of complex numbers with zero imaginary part
- **Trigonometry**: $\sin\theta$ and $\cos\theta$ connect to Euler's formula
- **Vectors in $\mathbb{R}^2$**: Complex addition mirrors vector addition
- **Polynomials**: Complex numbers guarantee roots of any polynomial
- **Linear Algebra**: Complex vector spaces and matrices

## Next Concepts

- **Complex Analysis**: Differentiability of complex functions, Cauchy-Riemann equations, contour integration
- **Fourier Analysis**: Decomposing signals into complex sinusoids
- **Quantum Computing**: Qubits, quantum gates, and measurement in $\mathbb{C}^n$
- **Complex-Valued Neural Networks**: CVNN architectures and backpropagation in the complex domain
- **Control Theory**: Transfer functions and the $s$-plane

## Summary

Complex numbers extend the real number line to a two-dimensional plane by introducing the imaginary unit $i$ where $i^2 = -1$. Every complex number has the form $a + bi$ and can be plotted on an Argand diagram. The modulus $|z| = \sqrt{a^2 + b^2}$ measures distance from the origin, and the conjugate $\bar{z} = a - bi$ reflects across the real axis. Arithmetic follows natural rules, with division requiring multiplication by the conjugate. Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ unifies trigonometry and exponentials, enabling polar representation. Complex numbers are indispensable in signal processing, quantum mechanics, control theory, and emerging AI fields like quantum ML and complex-valued neural networks.

## Key Takeaways

1. A complex number is $z = a + bi$ where $i^2 = -1$
2. The complex plane (Argand diagram) plots real vs. imaginary parts
3. The modulus $|z|$ is the distance from the origin; the conjugate $\bar{z}$ reflects across the real axis
4. Arithmetic follows standard algebra, replacing $i^2$ with $-1$
5. Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ connects exponentials to trigonometry
6. Any complex number has polar form $z = r e^{i\theta}$ useful for multiplication and powers
7. Complex numbers enable the solution of any polynomial equation
8. Applications span signal processing, quantum mechanics, control theory, and modern ML
