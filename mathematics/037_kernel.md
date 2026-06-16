# Concept: Kernel (Nullspace)

## Concept ID

MATH-037

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Define the kernel (nullspace) of a matrix and a linear transformation.
- Compute a basis for the kernel using Gaussian elimination and row reduction to reduced row echelon form (RREF).
- State and apply the Rank–Nullity Theorem to relate the dimensions of the domain, the image, and the kernel.
- Characterize injective (one-to-one) linear transformations via the triviality of the kernel.
- Relate the kernel to the solution set of a homogeneous system $A\mathbf{x} = \mathbf{0}$.
- Distinguish the linear-algebra notion of a kernel from the kernel trick / reproducing kernel Hilbert spaces used in support vector machines.
- Analyze the information loss incurred by a linear layer in a neural network by examining its nullspace.

## Prerequisites

- Matrix notation and basic matrix operations (addition, multiplication).
- Systems of linear equations and Gaussian elimination.
- Row reduction, pivot positions, and reduced row echelon form (RREF).
- Vector spaces and subspaces: definition, closure, span, linear independence, basis, dimension.
- The concept of a linear transformation and its matrix representation.
- Determinants for square matrices (basic properties).

## Definition

Let $A \in \mathbb{R}^{m \times n}$ be a real matrix with $m$ rows and $n$ columns. The **kernel** (also called the **nullspace**) of $A$ is the set of all vectors $\mathbf{x} \in \mathbb{R}^{n}$ such that $A\mathbf{x} = \mathbf{0}$.

\[
\ker(A) \;=\; \{\mathbf{x} \in \mathbb{R}^{n} \mid A\mathbf{x} = \mathbf{0}\}.
\]

If $T : \mathbb{R}^{n} \to \mathbb{R}^{m}$ is the linear transformation defined by $T(\mathbf{x}) = A\mathbf{x}$, then $\ker(T) = \ker(A)$. The **nullity** of $A$, denoted $\operatorname{nullity}(A)$, is the dimension of $\ker(A)$:

\[
\operatorname{nullity}(A) \;=\; \dim\bigl(\ker(A)\bigr).
\]

## Intuition

Imagine a linear transformation as a machine that takes vectors from $\mathbb{R}^{n}$ and produces vectors in $\mathbb{R}^{m}$. The kernel is the collection of all input vectors that the machine "crushes" down to the zero vector. If the kernel contains only the zero vector, the machine never sends two different inputs to the same output — every non-zero input produces something non-zero. If the kernel is larger (has positive dimension), the machine irretrievably loses information: any difference between two inputs that lies inside the kernel becomes invisible at the output.

Geometrically, think of a projection onto a plane in $\mathbb{R}^{3}$. The entire line of vectors orthogonal to the plane gets mapped to the origin — that line is the kernel. Every point on that line collapses to $\mathbf{0}$, so the three-dimensional space is "squashed" into a two-dimensional plane, and the information along the direction of the kernel is lost.

## Why This Concept Matters

The kernel is one of the four fundamental subspaces of a matrix and sits at the heart of linear algebra. It tells us:

- **Injectivity**: a linear map is one-to-one if and only if its kernel contains only $\mathbf{0}$.
- **Solvability**: the equation $A\mathbf{x} = \mathbf{b}$ has at most one solution iff $\ker(A) = \{\mathbf{0}\}$.
- **Solutions of homogeneous systems**: $\ker(A)$ is precisely the solution set of $A\mathbf{x} = \mathbf{0}$. This set is always a subspace, and a basis for it describes every possible solution.
- **Rank–Nullity**: the dimensions of the kernel and the image add up to the dimension of the domain, giving a fundamental conservation law of linear maps.

## Historical Background

The word "kernel" was introduced by the German mathematician **David Hilbert** (1862–1943) in his seminal work on integral equations around 1904. He used the German word *Kern* to mean the "core" or "nucleus" of an integral operator — the function that encodes the transformation. The term was later absorbed into abstract linear algebra to denote the set of vectors mapped to zero.

The related term "nullspace" has roots in the study of homogeneous linear equations. **Augustin-Louis Cauchy** and **Arthur Cayley** in the 19th century recognized that the solution set of $A\mathbf{x} = \mathbf{0}$ forms a linear space. The systematic treatment of nullspaces as subspaces, along with the rank–nullity theorem, was formalised in the early 20th century through the work of **Ferdinand Georg Frobenius**, **James Joseph Sylvester**, and later **Paul Halmos** and **Peter Lax**.

The Rank–Nullity Theorem is sometimes called the **dimension theorem** or the **first isomorphism theorem for vector spaces**. It is a special case of the more general isomorphism theorems in abstract algebra.

## Real World Examples

### 1. Computer Graphics and 3D Projections

When a 3D scene is rendered onto a 2D screen, the projection matrix $P$ maps $\mathbb{R}^{3}$ to $\mathbb{R}^{2}$. The kernel of $P$ consists of all vectors that lie along the camera's viewing direction — points that appear at the same pixel on the screen are indistinguishable, but more importantly, any vector parallel to the camera's optical axis is mapped to $\mathbf{0}$.

### 2. Structural Engineering — States of Self-Stress

In a truss or pin-jointed structure, the equilibrium matrix $A$ relates member forces to external loads. The kernel of $A$ corresponds to **states of self-stress** — internal force distributions that produce zero net external load. These are crucial for understanding prestressing and the stability of structures.

### 3. Electrical Circuits (Kirchhoff's Laws)

The incidence matrix of an electrical circuit encodes the topology of nodes and branches. The kernel of the incidence matrix (over an appropriate field) corresponds to cycles (loops) in the circuit — current distributions that satisfy Kirchhoff's current law with zero net external current.

### 4. Cryptography — Linear Codes

In coding theory, a linear code $C$ can be defined as the kernel of a parity-check matrix $H$. The code consists of all vectors $\mathbf{x}$ such that $H\mathbf{x} = \mathbf{0}$. The nullspace of $H$ is exactly the set of valid codewords.

### 5. Robotics — Singular Configurations

A robot manipulator's Jacobian matrix $J$ maps joint velocities to end-effector velocities. The kernel of $J$ at a given configuration contains the **joint velocities that produce zero end-effector velocity** — these are the ones that move the arm without moving the tip. When $\ker(J)$ is non-trivial (a "singular configuration"), the robot loses degrees of freedom and may become uncontrollable in certain directions.

## AI/ML Relevance

### 1. Linear Layers in Neural Networks

A fully-connected layer computes $\mathbf{y} = W\mathbf{x} + \mathbf{b}$. The kernel of the weight matrix $W$ reveals what input differences are "invisible" to the layer. If $\ker(W)$ is non-trivial, two distinct inputs $\mathbf{x}_1$ and $\mathbf{x}_2$ differing by an element of $\ker(W)$ produce identical outputs. This loss of information compounds through deep networks and relates directly to the **rank** of the weight matrix. Understanding the nullspace helps in network pruning, regularization, and diagnosing representational bottlenecks.

### 2. Dimensionality Reduction (PCA)

Principal Component Analysis projects data onto the top $k$ principal components. The projection matrix has a kernel consisting of the discarded minor components. The nullspace dimension ($n - k$) quantifies how much information is lost during the reduction.

### 3. Underdetermined Systems and Inverse Problems

In machine learning, many problems are underdetermined ($m < n$): there are fewer equations than unknowns. The kernel then has dimension at least $n - m$, and the nullspace vectors represent the "nullspace component" of the solution that cannot be determined from the data. Techniques such as $\ell_2$ regularization (Tikhonov) and $\ell_1$ regularization (LASSO) effectively select the particular solution from the affine solution space by penalizing different components of the kernel.

### 4. Kernel Methods in SVMs — Important Distinction

It is critical to distinguish the **linear-algebra kernel** from the **kernel trick** used in Support Vector Machines and Gaussian Processes. The latter arises from **Mercer's theorem** and **reproducing kernel Hilbert spaces** (RKHS). A Mercer kernel is a function $k(\mathbf{x}, \mathbf{x}')$ that defines an inner product in a (potentially infinite-dimensional) feature space. The two concepts share the name "kernel" but are fundamentally different: the linear-algebra kernel is a subspace, while the RKHS kernel is a similarity function. Their only connection is that the RKHS kernel appears as the kernel of an integral operator in functional analysis, tying back to Hilbert's original work.

### 5. Autoencoders and Representation Learning

In linear autoencoders (which learn to approximate the identity mapping), the encoder projects data into a low-dimensional latent space, and the decoder reconstructs it. The nullspace of the decoder corresponds to components of the latent code that are "forgotten" during reconstruction — a direct analogue of information loss.

## Mathematical Explanation

Let $A \in \mathbb{R}^{m \times n}$. The system $A\mathbf{x} = \mathbf{0}$ is a homogeneous system of $m$ linear equations in $n$ unknowns. The solution set is always a subspace of $\mathbb{R}^{n}$:

- **Closure under addition**: If $A\mathbf{x}_1 = \mathbf{0}$ and $A\mathbf{x}_2 = \mathbf{0}$, then $A(\mathbf{x}_1 + \mathbf{x}_2) = \mathbf{0}$.
- **Closure under scalar multiplication**: If $A\mathbf{x} = \mathbf{0}$, then $A(c\mathbf{x}) = c \cdot \mathbf{0} = \mathbf{0}$.
- **Contains zero**: $A\mathbf{0} = \mathbf{0}$.

To compute $\ker(A)$, perform Gaussian elimination on $A$ to obtain its reduced row echelon form (RREF). Identify pivot columns (linearly independent) and free columns (dependent). For each free variable, set it to $1$ and all other free variables to $0$, then solve for the pivot variables. The resulting vectors form a basis of $\ker(A)$. The number of vectors in this basis equals the number of free variables.

The **Rank–Nullity Theorem** states:

\[
\operatorname{rank}(A) + \operatorname{nullity}(A) = n,
\]

where $n$ is the number of columns of $A$ (the dimension of the domain). Here, $\operatorname{rank}(A)$ is the dimension of the column space of $A$ (the image), and $\operatorname{nullity}(A) = \dim(\ker(A))$.

For a square $n \times n$ matrix $A$, the following are equivalent:

- $\ker(A) = \{\mathbf{0}\}$.
- $A$ is invertible (non-singular).
- $\det(A) \neq 0$.
- The columns of $A$ are linearly independent.
- $\operatorname{nullity}(A) = 0$ and $\operatorname{rank}(A) = n$.

## Formula(s)

\[
\ker(A) = \{\mathbf{x} \in \mathbb{R}^{n} \mid A\mathbf{x} = \mathbf{0}\}
\]

\[
\operatorname{nullity}(A) = \dim\bigl(\ker(A)\bigr)
\]

\[
\operatorname{rank}(A) + \operatorname{nullity}(A) = n \qquad \text{(Rank–Nullity Theorem)}
\]

\[
\ker(A) = \{\mathbf{0}\} \iff A \text{ is injective}
\]

\[
A \in \mathbb{R}^{n \times n},\; \ker(A) = \{\mathbf{0}\} \iff \det(A) \neq 0 \iff A \text{ is invertible}
\]

## Properties

1. **Subspace**: $\ker(A)$ is always a subspace of $\mathbb{R}^{n}$ (the domain space).

2. **Injectivity**: $T(\mathbf{x}) = A\mathbf{x}$ is injective (one-to-one) iff $\ker(A) = \{\mathbf{0}\}$.

3. **Invertibility (square matrices)**: For $A \in \mathbb{R}^{n \times n}$, $\ker(A) = \{\mathbf{0}\}$ iff $A$ is invertible.

4. **Kernel of a product**: $\ker(AB) \supseteq \ker(B)$. In general, the kernel of a product is at least as large as the kernel of the right factor.

5. **Kernel and transpose**: $\ker(A) = \ker(A^{\mathsf{T}} A)$ for any real matrix $A$. This is a key fact used in least-squares problems.

6. **Orthogonality**: $\ker(A)$ is orthogonal to the row space of $A$. That is, every vector in the nullspace is orthogonal to every row of $A$.

7. **Overdetermined vs. underdetermined**: If $m > n$ (more rows than columns), the kernel may still be $\{\mathbf{0}\}$. If $m < n$ (fewer rows than columns), the kernel must have dimension at least $n - m > 0$, so $\ker(A) \neq \{\mathbf{0}\}$.

8. **Kernel of a zero matrix**: If $A = \mathbf{0}_{m \times n}$, then $\ker(A) = \mathbb{R}^{n}$ and $\operatorname{nullity}(A) = n$.

9. **Kernel of an identity matrix**: If $A = I_n$, then $\ker(A) = \{\mathbf{0}\}$.

10. **Similarity invariance**: If $B = P^{-1} A P$ for an invertible $P$, then $\ker(B) = P^{-1}\ker(A)$. The kernel is structurally preserved under similarity transformations.

## Step-by-Step Worked Examples

### Example 1: Finding the kernel of a $2 \times 3$ matrix

Find a basis for the kernel of $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$.

**Step 1**: Write the homogeneous system $A\mathbf{x} = \mathbf{0}$.

\[
\begin{cases}
x_1 + 2x_2 + 3x_3 = 0 \\
2x_1 + 4x_2 + 6x_3 = 0
\end{cases}
\]

**Step 2**: Row-reduce $A$ to RREF.

\[
\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}
\xrightarrow{R_2 \leftarrow R_2 - 2R_1}
\begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \end{bmatrix}
\]

The RREF has one pivot in column 1. Columns 2 and 3 are free.

**Step 3**: Express pivot variables in terms of free variables.

From the first row: $x_1 + 2x_2 + 3x_3 = 0 \implies x_1 = -2x_2 - 3x_3$.

**Step 4**: Construct basis vectors.

- Set $x_2 = 1$, $x_3 = 0$: $x_1 = -2$. Vector: $\mathbf{v}_1 = \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}$.
- Set $x_2 = 0$, $x_3 = 1$: $x_1 = -3$. Vector: $\mathbf{v}_2 = \begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}$.

**Step 5**: The kernel is the span of these two vectors.

\[
\ker(A) = \operatorname{span}\left\{
\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix},
\begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}
\right\}.
\]

Nullity $= 2$. Rank $= 1$. Rank–Nullity: $1 + 2 = 3 = n$, verified.

---

### Example 2: Trivial kernel (invertible matrix)

Find $\ker(A)$ for $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$.

**Step 1**: Solve $A\mathbf{x} = \mathbf{0}$.

\[
\begin{cases}
x_1 + 2x_2 = 0 \\
3x_1 + 4x_2 = 0
\end{cases}
\]

**Step 2**: Subtract $3$ times the first equation from the second:

\[
(3x_1 + 4x_2) - 3(x_1 + 2x_2) = 0 \implies -2x_2 = 0 \implies x_2 = 0.
\]

Then $x_1 = 0$.

**Step 3**: The only solution is $\mathbf{x} = \mathbf{0}$.

\[
\ker(A) = \{\mathbf{0}\}, \quad \operatorname{nullity}(A) = 0.
\]

Check: $\det(A) = 1\cdot 4 - 2\cdot 3 = -2 \neq 0$, confirming $A$ is invertible.

---

### Example 3: Kernel of a $3 \times 4$ matrix

Find a basis for $\ker(A)$ where $A = \begin{bmatrix} 1 & 2 & 0 & -1 \\ 0 & 1 & 1 & 2 \\ 1 & 3 & 1 & 1 \end{bmatrix}$.

**Step 1**: Row-reduce to RREF.

\[
\begin{bmatrix} 1 & 2 & 0 & -1 \\ 0 & 1 & 1 & 2 \\ 1 & 3 & 1 & 1 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_1}
\begin{bmatrix} 1 & 2 & 0 & -1 \\ 0 & 1 & 1 & 2 \\ 0 & 1 & 1 & 2 \end{bmatrix}
\]

\[
\xrightarrow{R_3 \leftarrow R_3 - R_2}
\begin{bmatrix} 1 & 2 & 0 & -1 \\ 0 & 1 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{bmatrix}
\]

\[
\xrightarrow{R_1 \leftarrow R_1 - 2R_2}
\begin{bmatrix} 1 & 0 & -2 & -5 \\ 0 & 1 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{bmatrix}
\]

Pivot columns: 1 and 2. Free columns: 3 and 4.

**Step 2**: Express pivot variables.

\[
x_1 - 2x_3 - 5x_4 = 0 \implies x_1 = 2x_3 + 5x_4
\]
\[
x_2 + x_3 + 2x_4 = 0 \implies x_2 = -x_3 - 2x_4
\]

**Step 3**: Construct basis vectors.

- $x_3 = 1, x_4 = 0$: $x_1 = 2$, $x_2 = -1$. Vector: $\mathbf{v}_1 = \begin{bmatrix} 2 \\ -1 \\ 1 \\ 0 \end{bmatrix}$.
- $x_3 = 0, x_4 = 1$: $x_1 = 5$, $x_2 = -2$. Vector: $\mathbf{v}_2 = \begin{bmatrix} 5 \\ -2 \\ 0 \\ 1 \end{bmatrix}$.

**Step 4**: Conclusion.

\[
\ker(A) = \operatorname{span}\left\{
\begin{bmatrix} 2 \\ -1 \\ 1 \\ 0 \end{bmatrix},
\begin{bmatrix} 5 \\ -2 \\ 0 \\ 1 \end{bmatrix}
\right\}.
\]

Nullity $= 2$. Rank (from pivots) $= 2$. Rank–Nullity: $2 + 2 = 4 = n$, verified.

---

### Example 4: Kernel of a projection matrix

Let $P = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$. This projects $\mathbb{R}^3$ onto the $xy$-plane. Find $\ker(P)$.

**Step 1**: Solve $P\mathbf{x} = \mathbf{0}$:

\[
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
= \begin{bmatrix} x_1 \\ x_2 \\ 0 \end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}.
\]

**Step 2**: This gives $x_1 = 0$, $x_2 = 0$, and $x_3$ is free.

**Step 3**: Basis: $\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$.

\[
\ker(P) = \operatorname{span}\left\{\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}\right\},
\quad \operatorname{nullity}(P) = 1.
\]

Geometrically, the entire $z$-axis is crushed to the origin.

## Visual Interpretation

Visualising the kernel is easiest in $\mathbb{R}^{2}$ and $\mathbb{R}^{3}$.

- **$2 \times 2$ invertible matrix**: The kernel is just $\{\mathbf{0}\}$. The transformation stretches, rotates, or shears the plane, but no non-zero vector maps to zero. Every vector is sent somewhere non-zero.

- **$2 \times 2$ singular matrix** (e.g., projection onto a line through the origin): Suppose $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$. This maps every vector $(x, y)$ to $(x, 0)$. The kernel is the $y$-axis: the line $\{(0, y)^{\mathsf{T}} \mid y \in \mathbb{R}\}$. Every vector on the $y$-axis collapses to the origin.

- **$3 \times 3$ projection onto a plane**: As in Example 4, the kernel is a line (the $z$-axis). The transformation squashes the entire 3D space onto the $xy$-plane, and the $z$-axis is the set of all vectors that are completely flattened.

- **$3 \times 3$ projection onto a line**: A matrix like $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$ has kernel = the $yz$-plane (a 2D subspace). The nullspace dimension is 2.

Think of the kernel as the "compression direction" of a linear map. If you imagine the transformation as a physical squeezing or collapsing, the kernel is the set of all points that are squeezed all the way down to the single point $\mathbf{0}$.

## Common Mistakes

1. **Assuming the kernel is always non-trivial**: Many students assume every matrix has a non-zero kernel. In fact, invertible matrices have $\ker(A) = \{\mathbf{0}\}$.

2. **Confusing the kernel with the column space**: The kernel lives in the **domain** $\mathbb{R}^{n}$, while the column space (image) lives in the **codomain** $\mathbb{R}^{m}$. They are subspaces of different ambient spaces and have no direct containment relation.

3. **Treating the kernel as a set of individual numbers**: The kernel is a set of **vectors**, not scalars. A common mistake is to write $\ker(A) = \{0\}$ when the correct notation is $\ker(A) = \{\mathbf{0}\}$.

4. **Forgetting that free variables produce basis vectors**: When computing the nullspace, each free variable gives one basis vector. Forgetting to include all free variables leads to an incomplete basis.

5. **Misapplying rank–nullity**: The rank–nullity theorem adds the rank (dimension of the image) to the nullity (dimension of the kernel) to get the number of **columns** $n$, not the number of rows $m$. A common error is to use $m$ instead of $n$.

6. **Thinking $\ker(A) = \ker(A^{\mathsf{T}})$**: In general, $\ker(A)$ and $\ker(A^{\mathsf{T}})$ are subspaces of different spaces ($\mathbb{R}^{n}$ vs. $\mathbb{R}^{m}$) and have no simple relationship. However, $\dim(\ker(A)) = \dim(\ker(A^{\mathsf{T}})) + (n - m)$ does not hold; instead, $\operatorname{rank}(A) = \operatorname{rank}(A^{\mathsf{T}})$.

7. **Assuming $\ker(AB) = \ker(B)$**: The kernel of a product is generally larger: $\ker(AB) \supseteq \ker(B)$. Equality holds only when $A$ is injective on the range of $B$.

8. **Confusing the solution set of $A\mathbf{x} = \mathbf{b}$ with the kernel**: The solution set of a non-homogeneous system is an **affine subspace** (a coset of the kernel), not a subspace itself. Only when $\mathbf{b} = \mathbf{0}$ is the solution set a subspace.

## Interview Questions

### Beginner

1. **Q**: What is the kernel of a matrix?
   **A**: The kernel (nullspace) of a matrix $A \in \mathbb{R}^{m \times n}$ is the set of all vectors $\mathbf{x} \in \mathbb{R}^{n}$ such that $A\mathbf{x} = \mathbf{0}$.

2. **Q**: How do you compute a basis for the nullspace of a matrix?
   **A**: Row-reduce $A$ to RREF, identify free columns, set each free variable to $1$ (others to $0$) one at a time, and solve for pivot variables.

3. **Q**: Is the kernel always a subspace? Why?
   **A**: Yes. If $A\mathbf{v} = \mathbf{0}$ and $A\mathbf{w} = \mathbf{0}$, then $A(\mathbf{v} + \mathbf{w}) = \mathbf{0}$ and $A(c\mathbf{v}) = c \cdot \mathbf{0} = \mathbf{0}$, so the kernel is closed under addition and scalar multiplication and contains $\mathbf{0}$.

4. **Q**: When is $\ker(A) = \{\mathbf{0}\}$ for a square matrix?
   **A**: When $A$ is invertible, i.e., when $\det(A) \neq 0$, or equivalently when the columns are linearly independent.

5. **Q**: What does it mean geometrically for a $3 \times 3$ matrix to have nullity $1$?
   **A**: It means the transformation crushes a 1-dimensional line of vectors to zero. The remaining two dimensions (the image) are spanned by the column space.

### Intermediate

1. **Q**: How does the kernel relate to the injectivity of a linear transformation?
   **A**: A linear transformation $T$ is injective iff $\ker(T) = \{\mathbf{0}\}$. If a non-zero vector maps to zero, that vector and $\mathbf{0}$ share the same image, violating one-to-one.

2. **Q**: For a real matrix $A$, prove that $\ker(A) = \ker(A^{\mathsf{T}} A)$.
   **A**: Clearly $\ker(A) \subseteq \ker(A^{\mathsf{T}} A)$ because if $A\mathbf{x} = \mathbf{0}$, then $A^{\mathsf{T}} A\mathbf{x} = \mathbf{0}$. Conversely, if $A^{\mathsf{T}} A\mathbf{x} = \mathbf{0}$, then $\mathbf{x}^{\mathsf{T}} A^{\mathsf{T}} A\mathbf{x} = \|A\mathbf{x}\|^2 = 0$, which implies $A\mathbf{x} = \mathbf{0}$. Hence the two kernels are equal.

3. **Q**: If $A$ is $m \times n$ with $m < n$, why must $\ker(A) \neq \{\mathbf{0}\}$?
   **A**: By rank–nullity, $\operatorname{rank}(A) + \operatorname{nullity}(A) = n$. Since $\operatorname{rank}(A) \leq m < n$, the nullity must be at least $n - m > 0$.

4. **Q**: What is the relationship between the kernel of $A$ and the linear dependence of its columns?
   **A**: The columns of $A$ are linearly independent iff $\ker(A) = \{\mathbf{0}\}$. A non-trivial kernel gives a non-trivial linear combination of columns equalling $\mathbf{0}$, i.e., a linear dependence among columns.

5. **Q**: Given $A\mathbf{x} = \mathbf{b}$, what can you say about the solution set in terms of the kernel?
   **A**: If $\mathbf{x}_p$ is one particular solution, then every solution is $\mathbf{x}_p + \mathbf{k}$ for some $\mathbf{k} \in \ker(A)$. The solution set is a coset of the kernel.

### Advanced

1. **Q**: In a deep neural network, how would you interpret the kernel of a weight matrix $W^{(l)}$ at layer $l$?
   **A**: The kernel of $W^{(l)}$ consists of all hidden representations at layer $l-1$ that produce a zero output at layer $l$. If the kernel is non-trivial, information is irretrievably lost atlayer $l$. Over multiple layers, the intersection of pre-images of these kernels can cause vanishing gradients and representational collapse. This connects to the notion of the **intrinsic dimension** of the learned representation.

2. **Q**: How does the kernel of the Jacobian in a neural network relate to the concept of a flat minimum in the loss landscape?
   **A**: At a local minimum of the loss, the Jacobian of the network outputs with respect to the parameters may have a non-trivial kernel. Directions in parameter space that lie in this kernel do not change the network output — they correspond to exactly flat directions in the loss landscape. The dimension of this kernel is related to the number of redundant parameters and the notion of **overparameterization**.

3. **Q**: In functional analysis, what is the relationship between the kernel of an integral operator and the kernel of its finite-dimensional approximation?
   **A**: For an integral operator $(Tf)(x) = \int_a^b K(x, y) f(y)\,dy$, the kernel in the linear-algebra sense is $\{f : Tf = 0\}$. When discretising the operator on a grid, we obtain a matrix $A$ whose kernel approximates the kernel of $T$, but the discretisation error and the choice of quadrature rule affect the accuracy. The study of how the nullspace of the discrete operator converges to that of the continuous operator is a key topic in numerical analysis.

## Practice Problems

### Easy

1. Find $\ker(A)$ and $\operatorname{nullity}(A)$ for $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$.

2. Find $\ker(A)$ and $\operatorname{nullity}(A)$ for $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$.

3. Determine $\operatorname{nullity}(A)$ for the $3 \times 5$ zero matrix.

4. For $A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$, find a basis for $\ker(A)$.

5. If $\operatorname{rank}(A) = 3$ for $A \in \mathbb{R}^{3 \times 7}$, what is $\operatorname{nullity}(A)$?

### Medium

1. Find a basis for $\ker(A)$ where $A = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 1 & 1 & 1 \end{bmatrix}$.

2. Prove that $\ker(A)$ is a subspace of $\mathbb{R}^{n}$.

3. Let $A$ and $B$ be $n \times n$ matrices. Show that $\ker(AB) \supseteq \ker(B)$. Give an example where the inclusion is strict.

4. For $A = \begin{bmatrix} 1 & -1 & 2 \\ 2 & 1 & 1 \\ -1 & 2 & 1 \end{bmatrix}$, find $\ker(A)$, $\operatorname{rank}(A)$, and verify the rank–nullity theorem.

5. If $\mathbf{v} \in \ker(A)$ and $\mathbf{w} \in \operatorname{rowspace}(A)$, prove $\mathbf{v} \perp \mathbf{w}$.

### Hard

1. Let $A \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^{m}$. Show that the general solution to $A\mathbf{x} = \mathbf{b}$ (if it exists) can be written as $\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h$, where $\mathbf{x}_p$ is any particular solution and $\mathbf{x}_h \in \ker(A)$. Use this to characterise when the solution is unique.

2. Consider the block matrix $M = \begin{bmatrix} A & B \\ 0 & C \end{bmatrix}$ with $A \in \mathbb{R}^{p \times q}$, $C \in \mathbb{R}^{r \times s}$. Describe $\ker(M)$ in terms of the kernels of $A$ and $C$.

3. For a symmetric matrix $S \in \mathbb{R}^{n \times n}$, prove that the kernel is orthogonal to the column space. Then show that $\mathbb{R}^{n} = \ker(S) \oplus \operatorname{col}(S)$.

## Solutions

### Easy — Solutions

**1.** $A = I_2$ gives $\ker(A) = \{\mathbf{0}\}$, $\operatorname{nullity}(A) = 0$.

**2.** $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$. One free variable: $x_2$. Equation: $x_1 + 2x_2 = 0 \implies x_1 = -2x_2$. Basis: $\begin{bmatrix} -2 \\ 1 \end{bmatrix}$. Nullity $= 1$, $\ker(A) = \operatorname{span}\{\begin{bmatrix} -2 \\ 1 \end{bmatrix}\}$.

**3.** Zero matrix maps every vector to $\mathbf{0}$, so $\ker(A) = \mathbb{R}^{5}$, $\operatorname{nullity}(A) = 5$.

**4.** $A$ has one pivot in column 1. Free variables: $x_2, x_3$. Equation: $x_1 = 0$, $x_2, x_3$ free. Basis: $\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$, $\mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$. $\ker(A) = \operatorname{span}\{\mathbf{e}_2, \mathbf{e}_3\}$.

**5.** Rank–Nullity: $\operatorname{nullity}(A) = n - \operatorname{rank}(A) = 7 - 3 = 4$.

### Medium — Solutions

**1.** Row-reduce $A$:

\[
\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 1 & 1 & 1 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_1}
\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 0 & 1 & -1 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_2}
\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 0 & 0 & 0 \end{bmatrix}.
\]

Pivot columns: 1, 2. Free column: 3. Equation: $x_1 + 2x_3 = 0 \implies x_1 = -2x_3$, $x_2 - x_3 = 0 \implies x_2 = x_3$. Basis: $\begin{bmatrix} -2 \\ 1 \\ 1 \end{bmatrix}$. Nullity $= 1$.

**2.** (i) $A\mathbf{0} = \mathbf{0}$, so $\mathbf{0} \in \ker(A)$. (ii) If $A\mathbf{v} = \mathbf{0}$ and $A\mathbf{w} = \mathbf{0}$, then $A(\mathbf{v} + \mathbf{w}) = \mathbf{0}$, so $\mathbf{v} + \mathbf{w} \in \ker(A)$. (iii) If $A\mathbf{v} = \mathbf{0}$, then $A(c\mathbf{v}) = c \cdot \mathbf{0} = \mathbf{0}$, so $c\mathbf{v} \in \ker(A)$. Hence $\ker(A)$ is a subspace.

**3.** If $\mathbf{x} \in \ker(B)$, then $B\mathbf{x} = \mathbf{0}$, so $AB\mathbf{x} = A\mathbf{0} = \mathbf{0}$, thus $\mathbf{x} \in \ker(AB)$. So $\ker(B) \subseteq \ker(AB)$. Strict example: $B = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$, $A = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$. Then $\ker(B) = \operatorname{span}\{\mathbf{e}_1\}$ but $\ker(AB) = \mathbb{R}^{2}$.

**4.** Compute RREF:

\[
A \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & -1 \\ 0 & 0 & 0 \end{bmatrix}.
\]

Pivots in cols 1, 2; free col 3. Equations: $x_1 + x_3 = 0 \implies x_1 = -x_3$, $x_2 - x_3 = 0 \implies x_2 = x_3$. Basis for kernel: $\begin{bmatrix} -1 \\ 1 \\ 1 \end{bmatrix}$. Nullity $= 1$. Rank $= 2$. $2 + 1 = 3 = n$, verified.

**5.** Let $\mathbf{v} \in \ker(A)$, so $A\mathbf{v} = \mathbf{0}$. Let $\mathbf{w} \in \operatorname{rowspace}(A)$, i.e., $\mathbf{w} = A^{\mathsf{T}} \mathbf{y}$ for some $\mathbf{y}$. Then $\mathbf{v}^{\mathsf{T}} \mathbf{w} = \mathbf{v}^{\mathsf{T}} A^{\mathsf{T}} \mathbf{y} = (A\mathbf{v})^{\mathsf{T}} \mathbf{y} = \mathbf{0}^{\mathsf{T}} \mathbf{y} = 0$. Hence $\mathbf{v} \perp \mathbf{w}$.

### Hard — Solutions

**1.** Let $\mathbf{x}_p$ be any particular solution: $A\mathbf{x}_p = \mathbf{b}$. For any $\mathbf{x}_h \in \ker(A)$, $A(\mathbf{x}_p + \mathbf{x}_h) = \mathbf{b} + \mathbf{0} = \mathbf{b}$. Conversely, if $\mathbf{x}$ and $\mathbf{x}'$ are two solutions, then $A(\mathbf{x} - \mathbf{x}') = \mathbf{b} - \mathbf{b} = \mathbf{0}$, so $\mathbf{x} - \mathbf{x}' \in \ker(A)$. The solution is unique iff $\ker(A) = \{\mathbf{0}\}$.

**2.** Let $M\mathbf{z} = \mathbf{0}$ where $\mathbf{z} = \begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix}$, $\mathbf{x} \in \mathbb{R}^{q}$, $\mathbf{y} \in \mathbb{R}^{s}$. Then:

\[
\begin{bmatrix} A & B \\ 0 & C \end{bmatrix}
\begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix}
= \begin{bmatrix} A\mathbf{x} + B\mathbf{y} \\ C\mathbf{y} \end{bmatrix}
= \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \end{bmatrix}.
\]

From the second block: $C\mathbf{y} = \mathbf{0}$, so $\mathbf{y} \in \ker(C)$. Then $A\mathbf{x} = -B\mathbf{y}$. The set of $\mathbf{x}$ satisfying this is $\mathbf{x}_p + \ker(A)$ where $\mathbf{x}_p$ is any solution to $A\mathbf{x}_p = -B\mathbf{y}$. So $\ker(M)$ is parameterised by $\mathbf{y} \in \ker(C)$ and the corresponding $\mathbf{x}$.

**3.** Since $S$ is symmetric, $\operatorname{col}(S) = \operatorname{row}(S)$. We already proved $\ker(S) \perp \operatorname{row}(S)$ in Problem Medium 5. Hence $\ker(S) \perp \operatorname{col}(S)$. By the rank–nullity theorem, $\dim(\ker(S)) + \dim(\operatorname{col}(S)) = n$, and since the two subspaces are orthogonal, they are complementary: $\mathbb{R}^{n} = \ker(S) \oplus \operatorname{col}(S)$.

## Related Concepts

- **Image (Column Space)**: The dual subspace to the kernel — the set of all reachable outputs.
- **Rank**: The dimension of the image; related to nullity via rank–nullity.
- **Row Space**: The span of the rows; orthogonal complement of the kernel.
- **Left Nullspace**: $\ker(A^{\mathsf{T}})$, the orthogonal complement of the column space.
- **Fundamental Theorem of Linear Algebra**: $\mathbb{R}^{n} = \ker(A) \oplus \operatorname{rowspace}(A)$ and $\mathbb{R}^{m} = \operatorname{colspace}(A) \oplus \ker(A^{\mathsf{T}})$.
- **Eigenvectors**: A non-zero vector $\mathbf{v}$ satisfying $A\mathbf{v} = \lambda \mathbf{v}$; when $\lambda = 0$, eigenvectors lie in $\ker(A)$.
- **Singular Value Decomposition (SVD)**: The right singular vectors corresponding to zero singular values form an orthonormal basis for $\ker(A)$.

## Next Concepts

- **Image (Column Space)** (MATH-038): The natural continuation — the reachable outputs of a linear transformation.
- **Left Nullspace** (MATH-039): The kernel of the transpose, orthogonal to the column space.
- **Singular Value Decomposition**: A factorisation that explicitly reveals the four fundamental subspaces.
- **Pseudoinverse**: The Moore–Penrose inverse, which selects the minimal-norm solution in the presence of a non-trivial kernel.
- **Fredholm Alternative**: A theorem about solvability of linear equations in infinite dimensions that generalises the kernel–image relationship.

## Summary

The kernel (nullspace) of a matrix $A$ is the set of all vectors $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$. It is always a subspace of the domain $\mathbb{R}^{n}$. Its dimension, called the nullity, satisfies the rank–nullity theorem: $\operatorname{rank}(A) + \operatorname{nullity}(A) = n$. The kernel captures exactly the information lost by the linear transformation; it is trivial ($\{\mathbf{0}\}$) precisely when $A$ is injective. Computing the kernel via row reduction is a fundamental skill, revealing free variables and the complete structure of the solution space of homogeneous systems. In machine learning, the kernel of weight matrices determines what information is lost in linear layers, and understanding the nullspace is essential for analysing underdetermined systems, regularisation, and representation learning. It is critical to distinguish this concept from the "kernel trick" used in SVMs and Gaussian Processes, which is a different mathematical object originating from functional analysis.

## Key Takeaways

- $\ker(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ is a subspace of $\mathbb{R}^{n}$.
- Nullity $= \dim(\ker(A))$; computed as the number of free variables in the RREF.
- Rank–Nullity Theorem: $\operatorname{rank}(A) + \operatorname{nullity}(A) = n$.
- $A$ is injective (one-to-one) iff $\ker(A) = \{\mathbf{0}\}$.
- For square matrices, $\ker(A) = \{\mathbf{0}\}$ iff $A$ is invertible ($\det A \neq 0$).
- The kernel reveals what information is lost in a linear transformation — crucial for understanding neural network layers and dimensionality reduction.
- Do **not** confuse the kernel (nullspace) with the kernel trick / RKHS kernel in SVMs.
- Every homogeneous system $A\mathbf{x} = \mathbf{0}$ has the kernel as its solution set; non-homogeneous solutions are cosets of the kernel.
