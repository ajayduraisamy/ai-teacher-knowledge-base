# Concept: Dot Product

## Concept ID

MATH-016

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Define the dot product of two vectors algebraically and geometrically
- Compute dot products in 2D and 3D
- Use the dot product to test orthogonality
- Apply the dot product to compute lengths, angles, and projections
- Understand the role of dot products in machine learning and AI

## Prerequisites

- Basic algebra and arithmetic
- Understanding of vectors as directed quantities with magnitude and direction
- Familiarity with vector notation $\mathbf{u} = \langle u_1, u_2 \rangle$ and $\mathbf{v} = \langle v_1, v_2 \rangle$
- Basic trigonometry: $\cos\theta$ for angles between $0^\circ$ and $180^\circ$

## Definition

The **dot product** (also called the scalar product or inner product) combines two vectors of equal dimension to produce a single scalar (a real number). There are two equivalent definitions:

**Algebraic definition:** For vectors $\mathbf{u} = \langle u_1, u_2, \dots, u_n \rangle$ and $\mathbf{v} = \langle v_1, v_2, \dots, v_n \rangle$,

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n
$$

**Geometric definition:** For vectors $\mathbf{u}$ and $\mathbf{v}$ with an angle $\theta$ between them,

$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta
$$

where $\|\mathbf{u}\|$ denotes the magnitude (length) of $\mathbf{u}$.

## Intuition

The dot product measures how much one vector "aligns" with another. When two vectors point in the same direction, the dot product is positive and large. When they are perpendicular, the dot product is zero. When they point in opposite directions, the dot product is negative.

Think of it like this: if you pull a box with a rope at an angle, the dot product tells you how much of your pulling force actually moves the box forward (the component parallel to the ground), versus how much is wasted pulling upward.

## Why This Concept Matters

The dot product is one of the most fundamental operations in linear algebra. It is the building block for:

- Computing lengths and distances
- Finding angles between vectors
- Testing whether vectors are perpendicular (orthogonal)
- Projecting one vector onto another
- Defining the cosine similarity used throughout machine learning
- Computing attention scores in transformer neural networks

Nearly every algorithm that works with vector data relies on the dot product at some level.

## Historical Background

The dot product was developed in the 19th century as part of vector analysis. The Irish mathematician **William Rowan Hamilton** (1805–1865) introduced quaternions, which contained the seeds of the dot and cross products. The American physicist **Josiah Willard Gibbs** (1839–1903) formalised modern vector notation and explicitly defined the dot product as we use it today. Gibbs's work in vector calculus became the standard language for physics and engineering, and later, for machine learning.

## Real World Examples

1. **Physics — Work:** The work $W$ done by a constant force $\mathbf{F}$ applied over a displacement $\mathbf{d}$ is $W = \mathbf{F} \cdot \mathbf{d}$. Only the component of force parallel to the displacement does work.
2. **Computer Graphics:** Dot products determine how much a surface is lit by a light source (Lambertian shading): $N \cdot L$, where $N$ is the surface normal and $L$ is the light direction.
3. **Navigation:** GPS and inertial navigation systems use dot products to compute relative positions and directions.
4. **Economics:** Portfolio correlation uses the dot product (through covariance) to measure how assets move together.
5. **Sports Analytics:** The dot product of a player's position vector and a movement vector quantifies their contribution to team positioning.

## AI/ML Relevance

The dot product is arguably the single most important operation in modern AI. Here are concrete examples:

1. **Attention Scores in Transformers:** The self-attention mechanism (used in GPT, BERT, Llama) computes attention scores as dot products between query and key vectors:
   $$
   \text{score}(Q_i, K_j) = Q_i \cdot K_j^T
   $$
   Higher dot product means higher attention — the model "focuses" more on tokens whose query and key vectors align.

2. **Linear Regression:** A linear model makes predictions as $y = \mathbf{w} \cdot \mathbf{x} + b$, where $\mathbf{w}$ is the weight vector and $\mathbf{x}$ is the input feature vector.

3. **Neural Network Layers:** Every fully-connected layer computes $\mathbf{W} \cdot \mathbf{x}$, which is a collection of dot products between each neuron's weight vector and the input.

4. **Similarity in NLP:** Word embeddings (Word2Vec, GloVe) use dot products to measure semantic similarity between words.

5. **Kernel Methods:** Support Vector Machines use dot products (via kernels) to compute similarity in high-dimensional feature spaces.

## Mathematical Explanation

The dot product of two vectors $\mathbf{u}$ and $\mathbf{v}$ can be understood in two complementary ways:

**Algebraic approach:** Multiply corresponding components and sum the results. For $\mathbf{u} = \langle 3, 4 \rangle$ and $\mathbf{v} = \langle 2, -1 \rangle$:
$$
\mathbf{u} \cdot \mathbf{v} = 3(2) + 4(-1) = 6 - 4 = 2
$$

**Geometric approach:** The dot product equals the product of the magnitudes times the cosine of the angle between them. This means:
- If $\theta = 0^\circ$ (vectors point same direction), $\cos\theta = 1$, so $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|$ (maximum positive)
- If $\theta = 90^\circ$ (vectors are perpendicular), $\cos\theta = 0$, so $\mathbf{u} \cdot \mathbf{v} = 0$
- If $\theta = 180^\circ$ (vectors point opposite), $\cos\theta = -1$, so $\mathbf{u} \cdot \mathbf{v} = -\|\mathbf{u}\|\|\mathbf{v}\|$ (maximum negative)

The two definitions are equivalent due to the law of cosines. The dot product is commutative, distributive over vector addition, and compatible with scalar multiplication.

## Formula(s)

**Algebraic formula (n-dimensions):**
$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
$$

**Geometric formula:**
$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
$$

**Relationship to magnitude:**
$$
\|\mathbf{u}\|^2 = \mathbf{u} \cdot \mathbf{u}
$$

**Angle from dot product:**
$$
\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

## Properties

1. **Commutativity:** $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
2. **Distributivity:** $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$
3. **Scalar multiplication:** $(c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v}) = \mathbf{u} \cdot (c\mathbf{v})$
4. **Orthogonality:** $\mathbf{u} \cdot \mathbf{v} = 0$ if and only if $\mathbf{u}$ is perpendicular to $\mathbf{v}$ (including the zero vector)
5. **Self-dot product:** $\mathbf{u} \cdot \mathbf{u} = \|\mathbf{u}\|^2 \geq 0$, with equality only when $\mathbf{u} = \mathbf{0}$
6. **Cauchy-Schwarz inequality:** $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$, with equality when vectors are collinear
7. **Not associative:** The dot product is defined only between vectors; $(\mathbf{u} \cdot \mathbf{v}) \cdot \mathbf{w}$ is not defined because $\mathbf{u} \cdot \mathbf{v}$ is a scalar

## Step-by-Step Worked Examples

### Example 1: Basic Dot Product in 2D

**Problem:** Compute the dot product of $\mathbf{u} = \langle 4, -3 \rangle$ and $\mathbf{v} = \langle 2, 5 \rangle$.

**Solution:**

**Step 1:** Identify the components:
- $u_1 = 4$, $u_2 = -3$
- $v_1 = 2$, $v_2 = 5$

**Step 2:** Apply the algebraic formula:
$$
\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 = (4)(2) + (-3)(5)
$$

**Step 3:** Compute:
$$
\mathbf{u} \cdot \mathbf{v} = 8 + (-15) = -7
$$

**Step 4:** Interpret: The dot product is negative, meaning the angle between $\mathbf{u}$ and $\mathbf{v}$ is greater than $90^\circ$.

### Example 2: Dot Product in 3D and Orthogonality Test

**Problem:** Are the vectors $\mathbf{u} = \langle 1, 2, -3 \rangle$ and $\mathbf{v} = \langle 4, -5, -2 \rangle$ orthogonal?

**Solution:**

**Step 1:** Compute the dot product:
$$
\mathbf{u} \cdot \mathbf{v} = (1)(4) + (2)(-5) + (-3)(-2)
$$

**Step 2:** Simplify:
$$
\mathbf{u} \cdot \mathbf{v} = 4 + (-10) + 6 = 0
$$

**Step 3:** Since $\mathbf{u} \cdot \mathbf{v} = 0$, the vectors are orthogonal (perpendicular).

### Example 3: Geometric Interpretation

**Problem:** Given $\|\mathbf{u}\| = 5$, $\|\mathbf{v}\| = 8$, and $\theta = 60^\circ$, find $\mathbf{u} \cdot \mathbf{v}$.

**Solution:**

**Step 1:** Recall the geometric formula:
$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
$$

**Step 2:** Substitute the known values:
$$
\mathbf{u} \cdot \mathbf{v} = (5)(8)\cos(60^\circ)
$$

**Step 3:** $\cos(60^\circ) = \frac{1}{2}$, so:
$$
\mathbf{u} \cdot \mathbf{v} = 40 \times \frac{1}{2} = 20
$$

**Step 4:** Verify with the algebraic approach (if we had components consistent with these magnitudes and angle, the sum of products would equal 20).

### Example 4: Computing Dot Product of Large Vectors

**Problem:** Let $\mathbf{u} = \langle 1, 0, -1, 2, 3 \rangle$ and $\mathbf{v} = \langle 2, -1, 0, 1, -2 \rangle$ be 5-dimensional vectors. Compute $\mathbf{u} \cdot \mathbf{v}$.

**Solution:**

**Step 1:** Pair up corresponding components:
$$
\mathbf{u} \cdot \mathbf{v} = (1)(2) + (0)(-1) + (-1)(0) + (2)(1) + (3)(-2)
$$

**Step 2:** Compute each term:
$$
= 2 + 0 + 0 + 2 + (-6)
$$

**Step 3:** Sum:
$$
= -2
$$

**Step 4:** Notice that this works for any dimension — the dot product always sums component-wise products.

### Example 5: Using Dot Product to Find a Missing Component

**Problem:** Find $k$ such that $\mathbf{u} = \langle 2, k, 4 \rangle$ is orthogonal to $\mathbf{v} = \langle 1, 3, -2 \rangle$.

**Solution:**

**Step 1:** Set the dot product equal to zero (orthogonality condition):
$$
\mathbf{u} \cdot \mathbf{v} = (2)(1) + (k)(3) + (4)(-2) = 0
$$

**Step 2:** Simplify:
$$
2 + 3k - 8 = 0
$$

**Step 3:** Solve for $k$:
$$
3k - 6 = 0 \implies 3k = 6 \implies k = 2
$$

**Step 4:** Check: $\mathbf{u} = \langle 2, 2, 4 \rangle$, $\mathbf{v} = \langle 1, 3, -2 \rangle$. Then $\mathbf{u} \cdot \mathbf{v} = 2 + 6 - 8 = 0$. The vectors are orthogonal.

## Visual Interpretation

Imagine two arrows starting from the same point. The dot product $\mathbf{u} \cdot \mathbf{v}$ equals the length of $\mathbf{u}$ multiplied by the length of the projection of $\mathbf{v}$ onto $\mathbf{u}$ (or vice versa).

Visually:
- If the arrows are nearly parallel, the "shadow" of one on the other is long, and the dot product is large.
- If the arrows are perpendicular, the shadow is a single point (zero length), and the dot product is zero.
- If the arrows point roughly opposite, the shadow falls on the opposite side, and the dot product is negative.

In 2D, you can picture the dot product as: $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \times (\text{signed length of projection of $\mathbf{v}$ onto $\mathbf{u}$})$.

## Common Mistakes

1. **Mixing up dot product and cross product:** The dot product produces a scalar; the cross product produces a vector. They are fundamentally different operations.

2. **Forgetting to sum all products:** The dot product is the **sum** of component-wise products, not just the list of products.

3. **Trying to dot vectors of different dimensions:** The dot product is only defined for vectors of the same length. $\langle 1, 2, 3 \rangle \cdot \langle 4, 5 \rangle$ is invalid.

4. **Confusing orthogonality with zero vectors:** The zero vector is orthogonal to every vector (since $\mathbf{0} \cdot \mathbf{v} = 0$), but this is a degenerate case often overlooked.

5. **Assuming dot product equals cosine similarity:** Dot product is not normalised; cosine similarity is $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$. A large dot product could mean vectors are long and closely aligned, or just that they are very long.

6. **Misapplying commutativity of scalars:** While $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$, the dot product is **not** associative — $(\mathbf{u} \cdot \mathbf{v}) \cdot \mathbf{w}$ is meaningless.

7. **Thinking the dot product always produces a positive number:** The dot product can be positive, zero, or negative depending on the angle between vectors.

## Interview Questions

### Beginner

1. What is the dot product of two vectors? How is it computed?
2. Compute $\langle 3, -2 \rangle \cdot \langle 1, 4 \rangle$.
3. What does it mean geometrically when the dot product of two non-zero vectors is zero?
4. Is the dot product commutative? Give a brief justification.
5. If $\mathbf{u} \cdot \mathbf{v} = 0$, what can you conclude about the vectors?

### Intermediate

1. Derive the relationship between the dot product and the cosine of the angle between two vectors.
2. Two vectors $\mathbf{u}$ and $\mathbf{v}$ satisfy $\mathbf{u} \cdot \mathbf{v} = -5$. If $\|\mathbf{u}\| = 3$ and $\|\mathbf{v}\| = 2$, what is the angle between them?
3. Show that $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 + 2(\mathbf{u} \cdot \mathbf{v})$.
4. How is the dot product used in the attention mechanism of transformer models?
5. Prove the Cauchy-Schwarz inequality: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$.

### Advanced

1. The Gram matrix $G$ of a set of vectors has entries $G_{ij} = \mathbf{v}_i \cdot \mathbf{v}_j$. Prove that $G$ is positive semidefinite.
2. In a Hilbert space, the inner product (a generalisation of the dot product) induces a norm. Show that the parallelogram law $\|\mathbf{u} + \mathbf{v}\|^2 + \|\mathbf{u} - \mathbf{v}\|^2 = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$ holds for dot-product-induced norms.
3. Explain how dot-product attention (scaled dot-product attention) differs from additive attention, and why dot-product attention is preferred in practice.

## Practice Problems

### Easy - 5 Questions

1. Compute $\langle 2, 3 \rangle \cdot \langle -1, 4 \rangle$.
2. Compute $\langle 1, 0, -2 \rangle \cdot \langle 3, 5, 1 \rangle$.
3. If $\|\mathbf{u}\| = 4$, $\|\mathbf{v}\| = 3$, and $\theta = 45^\circ$, find $\mathbf{u} \cdot \mathbf{v}$.
4. Are $\mathbf{u} = \langle 2, -1, 3 \rangle$ and $\mathbf{v} = \langle 1, 5, 1 \rangle$ orthogonal?
5. Simplify $(2\mathbf{u}) \cdot (3\mathbf{v})$ in terms of $\mathbf{u} \cdot \mathbf{v}$.

### Medium - 5 Questions

6. Find the angle between $\mathbf{u} = \langle 1, 3.2 \rangle$ and $\mathbf{v} = \langle 6.4, 0 \rangle$ (round to the nearest degree).
7. For what value of $k$ are $\mathbf{u} = \langle k, 2, -1 \rangle$ and $\mathbf{v} = \langle 1, 3, 2 \rangle$ orthogonal?
8. Given $\mathbf{u} \cdot \mathbf{v} = 12$, $\|\mathbf{u}\| = 3$, $\|\mathbf{v}\| = 5$, find $\cos\theta$ and $\theta$.
9. Show that $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$.
10. The vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal. If $\|\mathbf{u}\| = 6$ and $\|\mathbf{v}\| = 8$, find $\|\mathbf{u} + \mathbf{v}\|$.

### Hard - 3 Questions

11. Prove that for any vectors $\mathbf{u}, \mathbf{v}, \mathbf{w}$: $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$ using component notation.
12. Given $\mathbf{u} = \langle 1, 2, 2 \rangle$ and $\mathbf{v} = \langle -2, 1, 0 \rangle$, find a vector $\mathbf{w}$ that is orthogonal to both $\mathbf{u}$ and $\mathbf{v}$ (hint: consider the cross product, or solve a system).
13. Suppose $\mathbf{u}, \mathbf{v}, \mathbf{w}$ are unit vectors (magnitude 1) and $\mathbf{u} + \mathbf{v} + \mathbf{w} = \mathbf{0}$. Find $\mathbf{u} \cdot \mathbf{v} + \mathbf{v} \cdot \mathbf{w} + \mathbf{w} \cdot \mathbf{u}$.

## Solutions

### Easy Solutions

1. $\langle 2, 3 \rangle \cdot \langle -1, 4 \rangle = 2(-1) + 3(4) = -2 + 12 = 10$.
2. $\langle 1, 0, -2 \rangle \cdot \langle 3, 5, 1 \rangle = 1(3) + 0(5) + (-2)(1) = 3 + 0 - 2 = 1$.
3. $\mathbf{u} \cdot \mathbf{v} = (4)(3)\cos(45^\circ) = 12 \times \frac{\sqrt{2}}{2} = 6\sqrt{2}$.
4. $\langle 2, -1, 3 \rangle \cdot \langle 1, 5, 1 \rangle = 2(1) + (-1)(5) + 3(1) = 2 - 5 + 3 = 0$. Yes, they are orthogonal.
5. $(2\mathbf{u}) \cdot (3\mathbf{v}) = 2 \cdot 3 (\mathbf{u} \cdot \mathbf{v}) = 6(\mathbf{u} \cdot \mathbf{v})$.

### Medium Solutions

6. $\mathbf{u} \cdot \mathbf{v} = 1(6.4) + 3.2(0) = 6.4$. $\|\mathbf{u}\| = \sqrt{1^2 + 3.2^2} = \sqrt{1 + 10.24} = \sqrt{11.24} \approx 3.353$. $\|\mathbf{v}\| = 6.4$. $\cos\theta = \frac{6.4}{3.353 \times 6.4} = \frac{1}{3.353} \approx 0.2982$. $\theta \approx \cos^{-1}(0.2982) \approx 72.6^\circ \approx 73^\circ$.

7. Set dot product to zero: $k(1) + 2(3) + (-1)(2) = k + 6 - 2 = k + 4 = 0$. So $k = -4$.

8. $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \frac{12}{3 \times 5} = \frac{12}{15} = 0.8$. $\theta = \cos^{-1}(0.8) \approx 36.87^\circ$.

9. $\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v}) = \mathbf{u} \cdot \mathbf{u} - \mathbf{u} \cdot \mathbf{v} - \mathbf{v} \cdot \mathbf{u} + \mathbf{v} \cdot \mathbf{v} = \|\mathbf{u}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2$.

10. Since $\mathbf{u} \cdot \mathbf{v} = 0$: $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 + 2(0) = 36 + 64 = 100$. So $\|\mathbf{u} + \mathbf{v}\| = 10$.

### Hard Solutions

11. Let $\mathbf{u} = \langle u_1, \dots, u_n \rangle$, $\mathbf{v} = \langle v_1, \dots, v_n \rangle$, $\mathbf{w} = \langle w_1, \dots, w_n \rangle$.
    $$
    \mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \sum_{i=1}^n u_i (v_i + w_i) = \sum_{i=1}^n (u_i v_i + u_i w_i) = \sum_{i=1}^n u_i v_i + \sum_{i=1}^n u_i w_i = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}.
    $$

12. Let $\mathbf{w} = \langle w_1, w_2, w_3 \rangle$. Orthogonal to $\mathbf{u}$: $w_1 + 2w_2 + 2w_3 = 0$. Orthogonal to $\mathbf{v}$: $-2w_1 + w_2 + 0 = 0 \implies w_2 = 2w_1$. Substituting: $w_1 + 2(2w_1) + 2w_3 = 5w_1 + 2w_3 = 0 \implies w_3 = -2.5w_1$. Choose $w_1 = 2$ to avoid fractions: $\mathbf{w} = \langle 2, 4, -5 \rangle$. Check: $\mathbf{u} \cdot \mathbf{w} = 2 + 8 - 10 = 0$, $\mathbf{v} \cdot \mathbf{w} = -4 + 4 + 0 = 0$.

13. Take dot product of $\mathbf{u} + \mathbf{v} + \mathbf{w} = \mathbf{0}$ with itself:
    $$
    (\mathbf{u} + \mathbf{v} + \mathbf{w}) \cdot (\mathbf{u} + \mathbf{v} + \mathbf{w}) = 0
    $$
    Expanding: $\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 + \|\mathbf{w}\|^2 + 2(\mathbf{u} \cdot \mathbf{v} + \mathbf{v} \cdot \mathbf{w} + \mathbf{w} \cdot \mathbf{u}) = 0$.
    Since each is a unit vector, $\|\mathbf{u}\|^2 = \|\mathbf{v}\|^2 = \|\mathbf{w}\|^2 = 1$.
    $3 + 2S = 0$ where $S = \mathbf{u} \cdot \mathbf{v} + \mathbf{v} \cdot \mathbf{w} + \mathbf{w} \cdot \mathbf{u}$.
    So $S = -\frac{3}{2}$.

## Related Concepts

- **Cross Product** (MATH-017): Another product of vectors that produces a vector perpendicular to both inputs
- **Vector Projection** (MATH-018): Directly built on the dot product
- **Angle Between Vectors** (MATH-019): Uses the dot product to compute angles
- **Distance Between Vectors** (MATH-020): Uses norms derived from dot products
- **Vector Norms**: $\|\mathbf{u}\| = \sqrt{\mathbf{u} \cdot \mathbf{u}}$

## Next Concepts

- **014 Vector Norms**: Understanding magnitude
- **021 Linear Transformations**: How dot products define transformations
- **026 Eigenvalues and Eigenvectors**: Spectral theorem uses inner products
- **031 Gram-Schmidt Process**: Orthogonalisation using dot products

## Summary

The dot product is a fundamental operation that takes two equal-length vectors and returns a scalar. It can be computed algebraically (sum of component-wise products) or geometrically (product of magnitudes times cosine of angle). A zero dot product indicates orthogonal vectors. The dot product is commutative, distributive, and is the foundation for computing lengths, angles, projections, and similarity measures used throughout mathematics and AI.

## Key Takeaways

- $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$
- The dot product is a scalar, not a vector
- Two non-zero vectors are orthogonal iff their dot product is zero
- Dot products drive attention mechanisms in transformers, linear models in ML, and similarity search in vector databases
- The Cauchy-Schwarz inequality bounds the dot product: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$
