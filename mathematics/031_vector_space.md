# Concept: Vector Space

## Concept ID

MATH-031

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

1. Define a vector space over a field and verify all eight axioms for a given set
2. Distinguish between vector spaces and non-examples by checking axiom violations
3. Identify common vector spaces including $\mathbb{R}^n$, $\mathbb{R}^{m\times n}$, and polynomial spaces
4. Recognize subspaces and prove closure under addition and scalar multiplication
5. Connect vector space structure to feature spaces and function spaces in machine learning

## Prerequisites

- Basic set theory: set membership, set builder notation, Cartesian products
- Field properties of real numbers: associativity, commutativity, distributivity, existence of additive and multiplicative identities and inverses
- Elementary matrix notation $\mathbb{R}^{m\times n}$
- Polynomial notation and basic algebra

## Definition

A vector space over a field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$) is a set $V$ equipped with two operations:

- **Vector addition**: $+ : V \times V \to V$
- **Scalar multiplication**: $\cdot : \mathbb{F} \times V \to V$

such that for all vectors $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and all scalars $a, b \in \mathbb{F}$, the following eight axioms hold:

1. **Closure under addition**: $\mathbf{u} + \mathbf{v} \in V$
2. **Commutativity of addition**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
3. **Associativity of addition**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
4. **Additive identity**: There exists $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$
5. **Additive inverse**: For each $\mathbf{v} \in V$, there exists $-\mathbf{v} \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
6. **Closure under scalar multiplication**: $a\mathbf{v} \in V$
7. **Distributivity of scalar multiplication over vector addition**: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$
8. **Distributivity of scalar multiplication over scalar addition**: $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$
9. **Compatibility of scalar multiplication with field multiplication**: $a(b\mathbf{v}) = (ab)\mathbf{v}$
10. **Multiplicative identity**: $1\mathbf{v} = \mathbf{v}$

Some textbooks consolidate these into eight axioms by merging closure conditions with the operation definitions.

## Intuition

A vector space is any mathematical universe where you can add things together and scale them up or down, and all the algebraic rules you expect from ordinary arithmetic continue to hold. Think of a blank canvas: you can place arrows (vectors) on it, combine them tip-to-tail (addition), or stretch them (scalar multiplication). The axioms guarantee that no matter what strange objects populate $V$ — whether they are arrows in the plane, matrices, polynomials, or functions — these two fundamental operations behave in a consistent, predictable way.

The power of the vector space concept lies in its abstraction. Once you prove that a set of objects satisfies the eight axioms, every theorem proven about vector spaces automatically applies to those objects. You never need to reprove that $\mathbb{R}^n$ has a basis, that polynomial spaces have dimension $n+1$, or that function spaces admit linear transformations — the general theory handles it all.

## Why This Concept Matters

Vector spaces are the stage upon which all of linear algebra is performed. Linear transformations are maps between vector spaces; matrices are concrete representations of these maps once bases are chosen; eigenvalues, eigenvectors, inner products, and norms all presuppose an underlying vector space structure. Without vector spaces, linear algebra would be a disconnected collection of techniques rather than a unified theory.

In applied mathematics and machine learning, vector spaces model the data universe. Every feature vector in $\mathbb{R}^d$ lives in a vector space. Every linear regression prediction is a linear combination of columns, which amounts to an operation in the column space. Kernel methods implicitly operate in high-dimensional (sometimes infinite-dimensional) reproducing kernel Hilbert spaces. Understanding vector spaces is the foundation for understanding all of these concepts.

## Historical Background

The concept of a vector space emerged gradually over the late 19th and early 20th centuries. Giuseppe Peano gave the first abstract axiomatic definition in 1888 in his work *Calcolo Geometrico*, though his formulation was limited to real scalars. Hermann Grassmann's 1844 *Ausdehnungslehre* (Theory of Extension) anticipated many vector space ideas but was too ahead of its time to gain immediate acceptance. The modern definition, allowing arbitrary fields, was solidified in the 1920s by mathematicians such as Emmy Noether and Emil Artin as part of the broader abstract algebra movement. Noether's work was particularly influential in showing that vector spaces could be studied over any field, unifying linear algebra across different domains of mathematics.

## Real World Examples

1. **GPS coordinates**: A position expressed as (latitude, longitude, altitude) is a vector in $\mathbb{R}^3$. Adding two displacement vectors gives a net displacement.
2. **RGB color space**: A color represented by (red, green, blue) intensity values lives in $\mathbb{R}^3$. Scalar multiplication dims or brightens the color.
3. **Portfolio returns**: A portfolio of $n$ assets is a vector of weights in $\mathbb{R}^n$. The set of all feasible portfolios forms a vector space if short selling is allowed.
4. **Audio signals**: A digital audio recording sampled at $N$ time points is a vector in $\mathbb{R}^N$. Adding two audio signals corresponds to mixing sounds.
5. **Grayscale images**: An $m \times n$ grayscale image can be flattened to a vector in $\mathbb{R}^{mn}$, enabling linear algebra techniques for image compression and denoising.

## AI/ML Relevance

Vector spaces are the natural habitat of machine learning data and models.

**Feature vectors**: Every tabular dataset consists of feature vectors $x_i \in \mathbb{R}^d$. Nearest neighbor classification, logistic regression, support vector machines, and neural networks all operate on vectors in $\mathbb{R}^d$. The choice of feature space determines what relationships the model can capture.

**Reproducing Kernel Hilbert Spaces (RKHS)**: Kernel methods like the SVM implicitly map data into a high-dimensional (possibly infinite-dimensional) vector space called a reproducing kernel Hilbert space. The celebrated "kernel trick" computes inner products in this space without explicitly constructing the feature map. For example, the Gaussian kernel $K(x, y) = \exp(-\gamma\|x - y\|^2)$ corresponds to an inner product in an infinite-dimensional vector space of functions.

**Function spaces in Gaussian processes**: A Gaussian process defines a distribution over functions. The space of functions that the GP prior can realize is a vector space (specifically, the reproducing kernel Hilbert space associated with the kernel). Understanding vector spaces is essential for grasping how Gaussian processes interpolate data and quantify uncertainty.

**Word embeddings**: In NLP, words are mapped to dense vectors in $\mathbb{R}^{300}$ (e.g., Word2Vec, GloVe). These vectors live in a vector space where semantic relationships correspond to vector arithmetic: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$.

**Neural network layers**: A fully connected layer computes $\mathbf{h} = W\mathbf{x} + \mathbf{b}$, which is an affine transformation in a vector space. The layer's output space is a vector space whose dimension equals the number of hidden units.

## Mathematical Explanation

### The Eight Axioms in Detail

Let $V$ be a set and $\mathbb{F}$ be a field (for our purposes, $\mathbb{R}$). We define two operations:

- **Addition**: $+ : V \times V \to V$, written $\mathbf{u} + \mathbf{v}$
- **Scalar multiplication**: $\cdot : \mathbb{F} \times V \to V$, written $a\mathbf{v}$

The axioms are:

**A1 — Closure under addition**: For all $\mathbf{u}, \mathbf{v} \in V$, $\mathbf{u} + \mathbf{v} \in V$.

This ensures that addition is a well-defined binary operation on $V$. Without this, we could not guarantee that adding two vectors stays within the set.

**A2 — Commutativity of addition**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.

The order of addition does not matter. This matches our geometric intuition that tip-to-tail addition is commutative.

**A3 — Associativity of addition**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.

Grouping does not matter. This allows us to write $\mathbf{u} + \mathbf{v} + \mathbf{w}$ without parentheses.

**A4 — Additive identity**: There exists $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ for all $\mathbf{v} \in V$.

The zero vector does nothing under addition. It is unique in any vector space.

**A5 — Additive inverse**: For each $\mathbf{v} \in V$, there exists $-\mathbf{v} \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.

Every vector can be "undone" by adding its negative. This enables subtraction $\mathbf{u} - \mathbf{v} := \mathbf{u} + (-\mathbf{v})$.

**A6 — Closure under scalar multiplication**: For all $a \in \mathbb{F}$ and $\mathbf{v} \in V$, $a\mathbf{v} \in V$.

Scaling a vector stays within the space.

**A7 — Distributivity of scalar multiplication over vector addition**: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$.

Scalar multiplication distributes across vector addition.

**A8 — Distributivity of scalar multiplication over scalar addition**: $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$.

Scalar multiplication distributes across scalar addition.

**A9 — Compatibility of scalar multiplication**: $a(b\mathbf{v}) = (ab)\mathbf{v}$.

The order of successive scaling does not matter.

**A10 — Multiplicative identity**: $1\mathbf{v} = \mathbf{v}$ where $1$ is the multiplicative identity in $\mathbb{F}$.

Scaling by 1 leaves the vector unchanged.

### Why Axioms Matter

The axioms define a minimal set of algebraic rules from which all other properties of vector spaces can be derived. For example, from the axioms we can prove:

- The zero vector is unique
- Additive inverses are unique
- $0\mathbf{v} = \mathbf{0}$ for all $\mathbf{v}$
- $a\mathbf{0} = \mathbf{0}$ for all $a$
- $(-1)\mathbf{v} = -\mathbf{v}$

If a set with operations violates even one axiom, it is not a vector space. The axioms serve as a litmus test.

### Example: $\mathbb{R}^n$

The set $\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) \mid x_i \in \mathbb{R}\}$ with componentwise addition and scalar multiplication:

$$(x_1, \ldots, x_n) + (y_1, \ldots, y_n) = (x_1 + y_1, \ldots, x_n + y_n)$$
$$a(x_1, \ldots, x_n) = (ax_1, \ldots, ax_n)$$

This satisfies all ten axioms. The zero vector is $(0, \ldots, 0)$. The additive inverse of $(x_1, \ldots, x_n)$ is $(-x_1, \ldots, -x_n)$.

### Example: $\mathbb{R}^{m \times n}$

The set of all $m \times n$ matrices with real entries, with entrywise addition and scalar multiplication, is a vector space. The zero vector is the zero matrix. Additive inverses are the entrywise negatives. This space has dimension $mn$.

### Example: Polynomial Spaces

Let $\mathcal{P}_n$ be the set of all polynomials of degree at most $n$ with real coefficients:

$$\mathcal{P}_n = \{a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n \mid a_i \in \mathbb{R}\}$$

with the usual polynomial addition and scalar multiplication. This is a vector space of dimension $n + 1$. The zero vector is the zero polynomial $0 + 0x + \cdots + 0x^n$.

### Example: Function Spaces

Let $\mathcal{F}([0,1], \mathbb{R})$ be the set of all functions $f : [0, 1] \to \mathbb{R}$ with pointwise addition $(f + g)(x) = f(x) + g(x)$ and pointwise scalar multiplication $(af)(x) = a f(x)$. This is an infinite-dimensional vector space.

## Formula(s)

**Addition**:
$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n)$$

**Scalar multiplication**:
$$a\mathbf{v} = (a v_1, a v_2, \ldots, a v_n)$$

**Linear combination**:
$$a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \cdots + a_k\mathbf{v}_k$$

**Zero vector in $\mathbb{R}^n$**:
$$\mathbf{0} = (0, 0, \ldots, 0)$$

## Properties

1. **Uniqueness of zero vector**: The additive identity is unique.
2. **Uniqueness of additive inverse**: Each vector has exactly one additive inverse.
3. **Zero scalar multiplication**: $0 \cdot \mathbf{v} = \mathbf{0}$ for all $\mathbf{v}$.
4. **Scalar multiplication by zero vector**: $a \cdot \mathbf{0} = \mathbf{0}$ for all $a$.
5. **Negation**: $(-1) \cdot \mathbf{v} = -\mathbf{v}$.
6. **Cancellation**: If $\mathbf{u} + \mathbf{v} = \mathbf{u} + \mathbf{w}$, then $\mathbf{v} = \mathbf{w}$.
7. **A subspace $W \subseteq V$ is itself a vector space** if and only if it is closed under addition and scalar multiplication (and contains $\mathbf{0}$).
8. **The intersection of any collection of subspaces is a subspace**.
9. **The sum of two subspaces $U + W = \{\mathbf{u} + \mathbf{w} \mid \mathbf{u} \in U, \mathbf{w} \in W\}$ is a subspace**.
10. **Direct sum**: If $U \cap W = \{\mathbf{0}\}$, then $U + W$ is called a direct sum, written $U \oplus W$, and every element can be written uniquely as $\mathbf{u} + \mathbf{w}$.

## Step-by-Step Worked Examples

### Example 1: Proving $\mathbb{R}^2$ is a Vector Space

**Problem**: Verify that $\mathbb{R}^2$ with standard addition and scalar multiplication satisfies the vector space axioms.

**Solution**:

Let $\mathbf{u} = (u_1, u_2)$, $\mathbf{v} = (v_1, v_2)$, $\mathbf{w} = (w_1, w_2) \in \mathbb{R}^2$, and $a, b \in \mathbb{R}$.

**A1 (Closure under addition)**:
$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2)$$
Since $u_1 + v_1 \in \mathbb{R}$ and $u_2 + v_2 \in \mathbb{R}$, the result is in $\mathbb{R}^2$. ✓

**A2 (Commutativity)**:
$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2) = (v_1 + u_1, v_2 + u_2) = \mathbf{v} + \mathbf{u}$$
by commutativity of real addition. ✓

**A3 (Associativity)**:
$$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = ((u_1 + v_1) + w_1, (u_2 + v_2) + w_2)$$
$$= (u_1 + (v_1 + w_1), u_2 + (v_2 + w_2)) = \mathbf{u} + (\mathbf{v} + \mathbf{w})$$
by associativity of real addition. ✓

**A4 (Additive identity)**:
Let $\mathbf{0} = (0, 0)$. Then $\mathbf{v} + \mathbf{0} = (v_1 + 0, v_2 + 0) = (v_1, v_2) = \mathbf{v}$. ✓

**A5 (Additive inverse)**:
Let $-\mathbf{v} = (-v_1, -v_2)$. Then $\mathbf{v} + (-\mathbf{v}) = (v_1 - v_1, v_2 - v_2) = (0, 0) = \mathbf{0}$. ✓

**A6 (Closure under scalar multiplication)**:
$$a\mathbf{v} = (a v_1, a v_2)$$
Since $a v_1 \in \mathbb{R}$ and $a v_2 \in \mathbb{R}$, the result is in $\mathbb{R}^2$. ✓

**A7 (Distributivity over vector addition)**:
$$a(\mathbf{u} + \mathbf{v}) = a(u_1 + v_1, u_2 + v_2) = (a(u_1 + v_1), a(u_2 + v_2))$$
$$= (a u_1 + a v_1, a u_2 + a v_2) = a\mathbf{u} + a\mathbf{v}$$
by distributivity in $\mathbb{R}$. ✓

**A8 (Distributivity over scalar addition)**:
$$(a + b)\mathbf{v} = ((a + b)v_1, (a + b)v_2) = (a v_1 + b v_1, a v_2 + b v_2) = a\mathbf{v} + b\mathbf{v}$$
by distributivity in $\mathbb{R}$. ✓

**A9 (Compatibility)**:
$$a(b\mathbf{v}) = a(b v_1, b v_2) = (a b v_1, a b v_2) = (ab)\mathbf{v}$$
by associativity of real multiplication. ✓

**A10 (Identity)**:
$$1\mathbf{v} = (1 \cdot v_1, 1 \cdot v_2) = (v_1, v_2) = \mathbf{v}$$
Since $1$ is the multiplicative identity in $\mathbb{R}$. ✓

All axioms hold, so $\mathbb{R}^2$ is a vector space over $\mathbb{R}$.

### Example 2: Showing a Set is NOT a Vector Space

**Problem**: Let $V = \{(x, y) \in \mathbb{R}^2 \mid x \geq 0, y \geq 0\}$, the first quadrant. Is $V$ a vector space under standard addition and scalar multiplication?

**Solution**:

**A4 (Additive identity)**: $\mathbf{0} = (0, 0) \in V$ since $0 \geq 0$ and $0 \geq 0$. ✓

**A5 (Additive inverses)**: Take $\mathbf{v} = (1, 1) \in V$. Its additive inverse would be $-\mathbf{v} = (-1, -1)$. But $(-1, -1) \notin V$ because $-1 < 0$. Therefore A5 fails.

Since at least one axiom fails, $V$ is not a vector space. (A6 also fails: take $a = -1$, $\mathbf{v} = (1, 1)$; then $a\mathbf{v} = (-1, -1) \notin V$.)

**Conclusion**: The first quadrant is not a vector space because it lacks additive inverses and is not closed under scalar multiplication with negative scalars.

### Example 3: Polynomial Space $\mathcal{P}_2$

**Problem**: Verify that the set $\mathcal{P}_2 = \{a_0 + a_1 x + a_2 x^2 \mid a_0, a_1, a_2 \in \mathbb{R}\}$ with standard polynomial addition and scalar multiplication satisfies the vector space axioms.

**Solution**:

Let $p(x) = a_0 + a_1 x + a_2 x^2$, $q(x) = b_0 + b_1 x + b_2 x^2$, $r(x) = c_0 + c_1 x + c_2 x^2$, and $c \in \mathbb{R}$.

**A1**: $p(x) + q(x) = (a_0 + b_0) + (a_1 + b_1)x + (a_2 + b_2)x^2 \in \mathcal{P}_2$. ✓

**A2**: $p(x) + q(x) = (a_0 + b_0) + (a_1 + b_1)x + (a_2 + b_2)x^2 = q(x) + p(x)$. ✓

**A3**: $(p + q) + r = p + (q + r)$ follows from associativity of real addition. ✓

**A4**: The zero polynomial $0(x) = 0 + 0x + 0x^2$ satisfies $p(x) + 0(x) = p(x)$. ✓

**A5**: $(-p)(x) = -a_0 - a_1 x - a_2 x^2$ satisfies $p(x) + (-p)(x) = 0(x)$. ✓

**A6**: $c p(x) = c a_0 + c a_1 x + c a_2 x^2 \in \mathcal{P}_2$. ✓

**A7—A10** follow from distributivity, associativity, and identity in $\mathbb{R}$. ✓

Therefore $\mathcal{P}_2$ is a vector space over $\mathbb{R}$.

### Example 4: Checking if a Subset is a Subspace

**Problem**: Determine whether $W = \{(x, y, z) \in \mathbb{R}^3 \mid x + y + z = 0\}$ is a subspace of $\mathbb{R}^3$.

**Solution**:

We need to check three conditions: $\mathbf{0} \in W$, closure under addition, and closure under scalar multiplication.

**Zero vector**: $(0, 0, 0)$ satisfies $0 + 0 + 0 = 0$, so $\mathbf{0} \in W$. ✓

**Closure under addition**: Let $\mathbf{u} = (u_1, u_2, u_3)$, $\mathbf{v} = (v_1, v_2, v_3) \in W$.
Then $u_1 + u_2 + u_3 = 0$ and $v_1 + v_2 + v_3 = 0$.
Now $\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, u_3 + v_3)$.
Check: $(u_1 + v_1) + (u_2 + v_2) + (u_3 + v_3) = (u_1 + u_2 + u_3) + (v_1 + v_2 + v_3) = 0 + 0 = 0$.
Thus $\mathbf{u} + \mathbf{v} \in W$. ✓

**Closure under scalar multiplication**: Let $a \in \mathbb{R}$ and $\mathbf{v} = (v_1, v_2, v_3) \in W$. Then $a\mathbf{v} = (a v_1, a v_2, a v_3)$.
Check: $a v_1 + a v_2 + a v_3 = a(v_1 + v_2 + v_3) = a \cdot 0 = 0$.
Thus $a\mathbf{v} \in W$. ✓

All conditions satisfied, so $W$ is a subspace of $\mathbb{R}^3$.

### Example 5: The Set of All Matrices with Zero Trace

**Problem**: Show that the set $V = \{A \in \mathbb{R}^{2 \times 2} \mid \text{tr}(A) = 0\}$ is a vector space under matrix addition and scalar multiplication.

**Solution**:

Let $A, B \in V$ and $c \in \mathbb{R}$. Then $\text{tr}(A) = 0$ and $\text{tr}(B) = 0$.

**Zero matrix**: $0_{2 \times 2}$ has trace $0 + 0 = 0$, so $0 \in V$. ✓

**Closure under addition**: $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B) = 0 + 0 = 0$, so $A + B \in V$. ✓

**Closure under scalar multiplication**: $\text{tr}(cA) = c \cdot \text{tr}(A) = c \cdot 0 = 0$, so $cA \in V$. ✓

All other axioms follow from the vector space structure of $\mathbb{R}^{2 \times 2}$. Hence $V$ is a vector space (a subspace of $\mathbb{R}^{2 \times 2}$).

## Visual Interpretation

A vector space can be visualized geometrically when the dimension is 2 or 3. In $\mathbb{R}^2$, vectors are arrows from the origin. Addition is tip-to-tail placement: place the tail of $\mathbf{v}$ at the tip of $\mathbf{u}$, and the result is the arrow from the origin to the combined tip. Scalar multiplication stretches ($|a| > 1$) or shrinks ($|a| < 1$) the arrow, and flips direction if $a < 0$.

The entire $\mathbb{R}^2$ plane is a vector space. Any line through the origin is a subspace (a 1-dimensional vector space). Any line not through the origin is NOT a vector space (it lacks the zero vector and closure under scalar multiplication).

In higher dimensions, visualization becomes symbolic, but the algebraic structure remains identical. An infinite-dimensional space, like the space of all continuous functions $C[0, 1]$, cannot be visualized directly but behaves algebraically like $\mathbb{R}^n$ with infinitely many coordinates.

## Common Mistakes

1. **Confusing the field with the vector set**: The scalars come from a field $\mathbb{F}$, while the vectors are elements of $V$. They are different sets. For example, in $\mathbb{R}^n$, scalars are single real numbers, vectors are $n$-tuples.

2. **Forgetting to check closure**: A subset of a vector space is not automatically a subspace. You must verify closure under addition and scalar multiplication, and that the zero vector is present.

3. **Assuming all sets with addition are vector spaces**: The set $\mathbb{N}^2$ of pairs of natural numbers is not a vector space: additive inverses are missing and scaling by non-integer scalars is not closed.

4. **Confusing the zero vector with the scalar 0**: The zero vector $\mathbf{0}$ is an element of $V$, while $0$ is a scalar in $\mathbb{F}$. They are different objects, though $0\mathbf{v} = \mathbf{0}$ connects them.

5. **Incorrectly applying distributivity**: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ involves scalar multiplication and vector addition. Do not confuse this with $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$, which involves scalar addition.

6. **Believing a vector space must have a defined inner product or norm**: Vector spaces do not require an inner product or norm by definition. Those are additional structures (inner product spaces, normed spaces) built on top of vector spaces.

7. **Assuming all vector spaces are finite-dimensional**: Function spaces like $C[0, 1]$ are infinite-dimensional vector spaces. Many important spaces in machine learning (e.g., RKHS) are infinite-dimensional.

8. **Thinking $\mathbb{R}^2$ is a subspace of $\mathbb{R}^3$**: Strictly speaking, $\mathbb{R}^2$ is not a subset of $\mathbb{R}^3$ because $(x, y) \neq (x, y, 0)$. However, $\{(x, y, 0) \mid x, y \in \mathbb{R}\}$ is a subspace of $\mathbb{R}^3$ isomorphic to $\mathbb{R}^2$.

9. **Neglecting to specify the field**: A vector space is always defined over a specific field. $\mathbb{R}^n$ over $\mathbb{R}$ is a vector space; $\mathbb{R}^n$ over $\mathbb{Q}$ is also a vector space but with different properties (infinite-dimensional over $\mathbb{Q}$).

10. **Assuming subspaces must be proper**: Every vector space is a subspace of itself, and $\{\mathbf{0}\}$ is always a subspace. These are called the trivial subspaces.

## Interview Questions

### Beginner

**Q1**: What is a vector space? State the key axioms.

**A**: A vector space over a field $\mathbb{F}$ is a set $V$ with addition and scalar multiplication satisfying ten axioms: closure under addition, commutativity, associativity, additive identity, additive inverses, closure under scalar multiplication, two distributivity laws, compatibility of scalar multiplication, and multiplicative identity $1\mathbf{v} = \mathbf{v}$.

**Q2**: Is $\mathbb{R}^3$ a vector space? Justify briefly.

**A**: Yes. $\mathbb{R}^3$ with componentwise addition and scalar multiplication satisfies all vector space axioms. The zero vector is $(0,0,0)$, additive inverses are $(-x, -y, -z)$, and all algebraic properties follow from those of real numbers.

**Q3**: What is the zero vector in $\mathbb{R}^{2 \times 3}$?

**A**: The $2 \times 3$ zero matrix: $\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$.

**Q4**: Why is the set $\{(x, y) \mid x, y \in \mathbb{R}, x + y = 1\}$ not a vector space?

**A**: It does not contain the zero vector $(0,0)$ since $0 + 0 \neq 1$, and it is not closed under addition: $(1,0) + (0,1) = (1,1)$, but $1+1=2 \neq 1$.

**Q5**: Give an example of a subspace of $\mathbb{R}^3$.

**A**: The $xy$-plane $W = \{(x, y, 0) \mid x, y \in \mathbb{R}\}$ is a subspace. It contains $(0,0,0)$, is closed under addition, and closed under scalar multiplication.

### Intermediate

**Q1**: Show that the set of all symmetric $n \times n$ matrices is a subspace of $\mathbb{R}^{n \times n}$.

**A**: Let $S = \{A \in \mathbb{R}^{n \times n} \mid A^T = A\}$. The zero matrix is symmetric. If $A, B$ are symmetric, then $(A+B)^T = A^T + B^T = A + B$, so $A+B$ is symmetric. If $c \in \mathbb{R}$, $(cA)^T = c A^T = c A$, so $cA$ is symmetric. Thus $S$ is a subspace.

**Q2**: Prove that in any vector space $V$, $0\mathbf{v} = \mathbf{0}$ for all $\mathbf{v} \in V$.

**A**: $0\mathbf{v} = (0+0)\mathbf{v} = 0\mathbf{v} + 0\mathbf{v}$ (by A8). Adding $-0\mathbf{v}$ to both sides: $0\mathbf{v} + (-0\mathbf{v}) = (0\mathbf{v} + 0\mathbf{v}) + (-0\mathbf{v})$. LHS is $\mathbf{0}$ by A5. RHS is $0\mathbf{v} + (0\mathbf{v} + (-0\mathbf{v})) = 0\mathbf{v} + \mathbf{0} = 0\mathbf{v}$ by A3, A5, A4. Thus $\mathbf{0} = 0\mathbf{v}$.

**Q3**: Is the union of two subspaces always a subspace? Prove or give a counterexample.

**A**: No. Let $U = \{(x, 0) \mid x \in \mathbb{R}\}$ and $W = \{(0, y) \mid y \in \mathbb{R}\}$ in $\mathbb{R}^2$. Then $(1, 0) \in U$ and $(0, 1) \in W$, but $(1, 0) + (0, 1) = (1, 1) \notin U \cup W$. So $U \cup W$ is not closed under addition.

**Q4**: What is the dimension of the space of $3 \times 3$ skew-symmetric matrices?

**A**: Skew-symmetric means $A^T = -A$. Diagonal entries must be 0. There are $\frac{3(3-1)}{2} = 3$ independent off-diagonal entries. The space has dimension 3.

**Q5**: Show that $\mathcal{P}_n$ (polynomials of degree $\leq n$) is isomorphic to $\mathbb{R}^{n+1}$.

**A**: The map $\phi: \mathcal{P}_n \to \mathbb{R}^{n+1}$ given by $\phi(a_0 + a_1 x + \cdots + a_n x^n) = (a_0, a_1, \ldots, a_n)$ is linear ($\phi(p+q) = \phi(p) + \phi(q)$, $\phi(cp) = c \phi(p)$), injective (zero polynomial maps to zero vector), and surjective (every $(a_0, \ldots, a_n)$ corresponds to some polynomial). Hence it's an isomorphism.

### Advanced

**Q1**: Prove that $C[0, 1]$, the set of continuous real-valued functions on $[0, 1]$, is an infinite-dimensional vector space.

**A**: $C[0, 1]$ is a vector space under pointwise addition and scalar multiplication. It is infinite-dimensional because the monomials $\{1, x, x^2, x^3, \ldots\}$ are linearly independent (a finite linear combination that is identically zero implies all coefficients are zero by the fundamental theorem of algebra). Since there are infinitely many independent vectors, $\dim(C[0, 1]) = \infty$.

**Q2**: Let $V$ be a vector space over a field $\mathbb{F}$. Prove that if $\mathbf{u}, \mathbf{v} \in V$ and $a \in \mathbb{F}$ satisfy $a\mathbf{u} = a\mathbf{v}$ and $a \neq 0$, then $\mathbf{u} = \mathbf{v}$.

**A**: If $a\mathbf{u} = a\mathbf{v}$, then $a\mathbf{u} - a\mathbf{v} = \mathbf{0}$, so $a(\mathbf{u} - \mathbf{v}) = \mathbf{0}$. Since $a \neq 0$, multiply both sides by $a^{-1}$ (which exists in $\mathbb{F}$): $a^{-1}[a(\mathbf{u} - \mathbf{v})] = a^{-1}\mathbf{0}$. LHS = $(a^{-1}a)(\mathbf{u} - \mathbf{v}) = 1(\mathbf{u} - \mathbf{v}) = \mathbf{u} - \mathbf{v}$ by A9 and A10. RHS = $\mathbf{0}$ by property 4. Thus $\mathbf{u} - \mathbf{v} = \mathbf{0}$, so $\mathbf{u} = \mathbf{v}$.

**Q3**: Explain how $\mathbb{R}^n$ as a feature space enables linear separability in SVMs, and how mapping to a higher-dimensional vector space (RKHS) can make non-separable data separable. Give a concrete data configuration.

**A**: In an SVM, data points $\mathbf{x}_i \in \mathbb{R}^d$ are classified by a hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$. If the data is not linearly separable in $\mathbb{R}^d$, a feature map $\phi: \mathbb{R}^d \to \mathcal{H}$ sends data into a higher-dimensional RKHS where a separating hyperplane exists. For example, XOR data: $\{(0,0), (1,1)\}$ in class $+1$, $\{(0,1), (1,0)\}$ in class $-1$ is not linearly separable in $\mathbb{R}^2$. Using the feature map $\phi(x_1, x_2) = (x_1, x_2, x_1 x_2)$, the points become $(0,0,0)$, $(1,1,1)$ for $+1$ and $(0,1,0)$, $(1,0,0)$ for $-1$, which are separable by the plane $x_3 = 0.5$. The kernel trick computes $K(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$ without explicitly constructing $\phi$.

## Practice Problems

### Easy — 5 Questions

**E1**: Determine whether $W = \{(x, y, z) \in \mathbb{R}^3 \mid x = 0\}$ is a subspace of $\mathbb{R}^3$.

**E2**: Is the set of all $2 \times 2$ matrices of the form $\begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix}$ a vector space?

**E3**: What is the zero vector in the space of all functions $f: \mathbb{R} \to \mathbb{R}$?

**E4**: Find the additive inverse of $p(x) = 3 - 2x + x^2$ in $\mathcal{P}_2$.

**E5**: Is the set $\{(x, y) \in \mathbb{R}^2 \mid y = 2x\}$ a subspace of $\mathbb{R}^2$?

### Medium — 5 Questions

**M1**: Prove that the intersection of two subspaces of a vector space is always a subspace.

**M2**: Show whether $W = \{A \in \mathbb{R}^{2 \times 2} \mid \det(A) = 0\}$ is a subspace of $\mathbb{R}^{2 \times 2}$.

**M3**: Determine whether $V = \{f: \mathbb{R} \to \mathbb{R} \mid f(0) = 0\}$ is a vector space under pointwise operations.

**M4**: Let $V$ be a vector space. Prove that if $\mathbf{v} + \mathbf{v} = \mathbf{v}$, then $\mathbf{v} = \mathbf{0}$.

**M5**: Is the set of all polynomials of degree exactly $3$ (not $\leq 3$) a vector space? Justify.

### Hard — 3 Questions

**H1**: Let $V$ be the set of all real sequences $(x_1, x_2, x_3, \ldots)$ that satisfy $x_{n+2} = x_{n+1} + x_n$. Show that $V$ is a vector space and find its dimension.

**H2**: Prove that in any vector space, $(-a)\mathbf{v} = -(a\mathbf{v})$ for any scalar $a$ and vector $\mathbf{v}$.

**H3**: Let $U$ and $W$ be subspaces of $V$. Prove that $U \cup W$ is a subspace if and only if $U \subseteq W$ or $W \subseteq U$.

## Solutions

### Easy Solutions

**E1**: Let $W = \{(x, y, z) \mid x = 0\}$. Then $(0, 0, 0) \in W$ (since $0 = 0$). If $\mathbf{u} = (0, u_2, u_3)$ and $\mathbf{v} = (0, v_2, v_3)$, then $\mathbf{u} + \mathbf{v} = (0, u_2 + v_2, u_3 + v_3) \in W$. For scalar $a$, $a\mathbf{u} = (0, a u_2, a u_3) \in W$. So $W$ is a subspace.

**E2**: Yes. The zero matrix $\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ has the form $\begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix}$ with $a = b = 0$. Sum of two such matrices is $\begin{pmatrix} a_1 + a_2 & 0 \\ 0 & b_1 + b_2 \end{pmatrix}$, which has the same form. Scalar multiple $c\begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix} = \begin{pmatrix} ca & 0 \\ 0 & cb \end{pmatrix}$ also has the form. All other axioms follow from $\mathbb{R}^{2 \times 2}$ properties. So it is a vector space (subspace of $\mathbb{R}^{2 \times 2}$).

**E3**: The zero function $z(x) = 0$ for all $x \in \mathbb{R}$. Check: $(f + z)(x) = f(x) + 0 = f(x)$.

**E4**: $-p(x) = -3 + 2x - x^2$. Check: $(3 - 2x + x^2) + (-3 + 2x - x^2) = 0$.

**E5**: Yes. Let $W = \{(x, 2x) \mid x \in \mathbb{R}\}$. Zero: $(0, 0) = (0, 2 \cdot 0) \in W$. Sum: $(x_1, 2x_1) + (x_2, 2x_2) = (x_1 + x_2, 2(x_1 + x_2)) \in W$. Scalar: $a(x, 2x) = (ax, 2ax) \in W$. So $W$ is a subspace of $\mathbb{R}^2$ (a line through origin).

### Medium Solutions

**M1**: Let $U$ and $W$ be subspaces. Show $U \cap W$ is a subspace.

- $\mathbf{0} \in U$ and $\mathbf{0} \in W$, so $\mathbf{0} \in U \cap W$.
- $\mathbf{v}, \mathbf{w} \in U \cap W$ implies $\mathbf{v}, \mathbf{w} \in U$ and $\mathbf{v}, \mathbf{w} \in W$. Since $U$ and $W$ are subspaces, $\mathbf{v} + \mathbf{w} \in U$ and $\mathbf{v} + \mathbf{w} \in W$, so $\mathbf{v} + \mathbf{w} \in U \cap W$.
- $\mathbf{v} \in U \cap W$ implies $\mathbf{v} \in U$ and $\mathbf{v} \in W$. For any scalar $a$, $a\mathbf{v} \in U$ and $a\mathbf{v} \in W$, so $a\mathbf{v} \in U \cap W$.

Hence $U \cap W$ is a subspace.

**M2**: $W = \{A \in \mathbb{R}^{2 \times 2} \mid \det(A) = 0\}$ is NOT a subspace. Counterexample: $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ has $\det = 0$, $B = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ has $\det = 0$, but $A + B = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ has $\det = 1 \neq 0$. Not closed under addition.

**M3**: Yes, $V$ is a vector space. The zero function $z(x) = 0$ satisfies $z(0) = 0$. If $f(0) = 0$ and $g(0) = 0$, then $(f+g)(0) = f(0) + g(0) = 0 + 0 = 0$, so $f+g \in V$. For scalar $a$, $(af)(0) = a \cdot f(0) = 0$, so $af \in V$. All other axioms follow from function space properties.

**M4**: If $\mathbf{v} + \mathbf{v} = \mathbf{v}$, add $-\mathbf{v}$ to both sides:
$(\mathbf{v} + \mathbf{v}) + (-\mathbf{v}) = \mathbf{v} + (-\mathbf{v})$
$\mathbf{v} + (\mathbf{v} + (-\mathbf{v})) = \mathbf{0}$ (by associativity)
$\mathbf{v} + \mathbf{0} = \mathbf{0}$
$\mathbf{v} = \mathbf{0}$

**M5**: No. Let $p(x) = x^3$ and $q(x) = -x^3 + x^2$. Both have degree exactly 3. But $p(x) + q(x) = x^2$, which has degree 2, not 3. Not closed under addition. Also, $0 \cdot p(x) = 0$ is the zero polynomial, which does not have degree 3. So it is not a vector space.

### Hard Solutions

**H1**: Let $V = \{(x_1, x_2, x_3, \ldots) \mid x_{n+2} = x_{n+1} + x_n \text{ for all } n \geq 1\}$.

- The zero sequence $(0, 0, 0, \ldots)$ satisfies $0 = 0 + 0$, so $\mathbf{0} \in V$.
- If $\mathbf{x}, \mathbf{y} \in V$, then $x_{n+2} = x_{n+1} + x_n$ and $y_{n+2} = y_{n+1} + y_n$. Then $(x+y)_{n+2} = x_{n+2} + y_{n+2} = (x_{n+1} + y_{n+1}) + (x_n + y_n) = (x+y)_{n+1} + (x+y)_n$. So $\mathbf{x} + \mathbf{y} \in V$.
- If $c \in \mathbb{R}$, then $(c x)_{n+2} = c x_{n+2} = c(x_{n+1} + x_n) = c x_{n+1} + c x_n = (c x)_{n+1} + (c x)_n$. So $c\mathbf{x} \in V$.

Thus $V$ is a vector space. To find dimension: every sequence in $V$ is determined by its first two terms $(x_1, x_2)$, because $x_3 = x_2 + x_1$, $x_4 = x_3 + x_2 = (x_2 + x_1) + x_2 = x_1 + 2x_2$, etc. So $\dim(V) = 2$.

**H2**: We need to show $(-a)\mathbf{v} + (a\mathbf{v}) = \mathbf{0}$.

$(-a)\mathbf{v} + a\mathbf{v} = ((-a) + a)\mathbf{v}$ (by A8) $= 0\mathbf{v} = \mathbf{0}$ (by property 3 of Properties section).

Thus $(-a)\mathbf{v}$ is the additive inverse of $a\mathbf{v}$, i.e., $(-a)\mathbf{v} = -(a\mathbf{v})$.

**H3**: ($\Leftarrow$) If $U \subseteq W$, then $U \cup W = W$, which is a subspace. Similarly if $W \subseteq U$.

($\Rightarrow$) Suppose $U \not\subseteq W$ and $W \not\subseteq U$. Then there exists $\mathbf{u} \in U \setminus W$ and $\mathbf{w} \in W \setminus U$. Consider $\mathbf{u} + \mathbf{w}$. Since $\mathbf{u} \in U$ and $\mathbf{w} \in W$, if $U \cup W$ were a subspace, then $\mathbf{u} + \mathbf{w} \in U \cup W$.

Case 1: $\mathbf{u} + \mathbf{w} \in U$. Then $\mathbf{w} = (\mathbf{u} + \mathbf{w}) - \mathbf{u} \in U$ (since $U$ is a subspace). Contradiction since $\mathbf{w} \notin U$.

Case 2: $\mathbf{u} + \mathbf{w} \in W$. Then $\mathbf{u} = (\mathbf{u} + \mathbf{w}) - \mathbf{w} \in W$. Contradiction since $\mathbf{u} \notin W$.

Thus $\mathbf{u} + \mathbf{w} \notin U \cup W$, so $U \cup W$ is not a subspace. Therefore, if $U \cup W$ is a subspace, then $U \subseteq W$ or $W \subseteq U$.

## Related Concepts

- **Field**: The scalar set over which a vector space is defined. Common fields are $\mathbb{R}$ (real numbers) and $\mathbb{C}$ (complex numbers).
- **Subspace**: A subset of a vector space that is itself a vector space under the same operations.
- **Linear combination**: An expression $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k$.
- **Linear independence**: A set of vectors where no vector is a linear combination of the others.
- **Span**: The set of all linear combinations of a set of vectors. Span is always a subspace.
- **Basis**: A linearly independent spanning set. Every vector space has a basis.
- **Dimension**: The number of vectors in any basis for a finite-dimensional vector space.
- **Linear transformation**: A map $T: V \to W$ between vector spaces that preserves addition and scalar multiplication.

## Next Concepts

- Basis and dimension (MATH-032)
- Span (MATH-033)
- Linear transformations and matrix representations
- Eigenvalues and eigenvectors
- Inner product spaces and norms
- Orthogonal projections and least squares
- Singular value decomposition

## Summary

A vector space is an abstract algebraic structure consisting of a set of vectors equipped with addition and scalar multiplication operations that satisfy a specific set of ten axioms (or eight, depending on how closure is counted). The axioms guarantee that vector addition is commutative, associative, has an identity and inverses, and that scalar multiplication distributes appropriately. Common examples include $\mathbb{R}^n$, $\mathbb{R}^{m \times n}$, polynomial spaces $\mathcal{P}_n$, and function spaces. Subspaces are subsets that are themselves vector spaces. The vector space axioms form the foundation of all of linear algebra — every theorem about bases, dimension, linear transformations, and eigenvalues ultimately relies on them.

## Key Takeaways

1. A vector space is set $V$ over a field $\mathbb{F}$ with addition and scalar multiplication satisfying ten axioms.
2. The axioms ensure algebraic consistency: addition is commutative and associative, there is a zero vector, every vector has a negative, and scalar multiplication distributes.
3. Every vector space contains the zero vector and is closed under addition and scalar multiplication.
4. $\mathbb{R}^n$, $\mathbb{R}^{m \times n}$, $\mathcal{P}_n$, and function spaces are all vector spaces.
5. A subset $W \subseteq V$ is a subspace if it contains $\mathbf{0}$ and is closed under addition and scalar multiplication.
6. Vector spaces can be finite-dimensional ($\mathbb{R}^n$) or infinite-dimensional (function spaces).
7. In machine learning, feature spaces, kernel spaces, and neural network output spaces are vector spaces.
8. Every vector space has a basis and a well-defined dimension.
9. The vector space framework unifies seemingly different mathematical objects under a single set of rules.
10. Understanding vector spaces is essential for advanced topics in linear algebra, functional analysis, and machine learning theory.
