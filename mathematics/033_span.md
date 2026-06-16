# Concept: Span

## Concept ID

MATH-033

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

1. Define the span of a set of vectors as the set of all linear combinations
2. Prove that the span of any set of vectors is a subspace
3. Determine whether a given vector lies in the span of a set
4. Connect the concept of span to column space, row space, and nullspace of a matrix
5. Relate span to basis: a basis is a spanning set that is also linearly independent
6. Apply span to understand feature spaces, column spaces, and solution spaces in machine learning

## Prerequisites

- Vector spaces and subspaces (MATH-031)
- Linear combinations: expressions of the form $a_1\mathbf{v}_1 + a_2\mathbf{v}_2 + \cdots + a_k\mathbf{v}_k$
- Systems of linear equations and Gaussian elimination
- Matrix-vector multiplication

## Definition

Let $S = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ be a set of vectors in a vector space $V$ over a field $\mathbb{F}$. The **span** of $S$, denoted $\text{span}(S)$ or $\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)$, is the set of all **finite linear combinations** of vectors in $S$:

$$\text{span}(S) = \left\{ \sum_{i=1}^{k} a_i \mathbf{v}_i \;\middle|\; a_i \in \mathbb{F} \right\}$$

If $S = \emptyset$ (the empty set), $\text{span}(\emptyset) = \{\mathbf{0}\}$ by convention.

The span of $S$ is always a **subspace** of $V$. It is the smallest subspace of $V$ that contains every vector in $S$.

A set $S$ is said to **span** $V$ (or be a **spanning set** for $V$) if $\text{span}(S) = V$.

## Intuition

Think of the span as the set of all points reachable by traveling along the vectors in $S$ and combining them in any proportion. If you have two non-collinear arrows in the plane, their span is the entire plane — you can reach any point by some combination of these two directions. If the arrows are collinear (one is a multiple of the other), their span is only the line through the origin in that direction.

The span answers the question: "What space do these vectors generate?" Three vectors in $\mathbb{R}^3$ can span the entire space (if they are not coplanar), a plane (if one is redundant), a line (if all are collinear), or just the origin (if all are zero).

Geometrically, adding a new vector to a set can either expand the span (if the new vector is not already in the span of the existing set) or leave it unchanged (if the new vector is redundant). A basis is a minimal spanning set — a set whose span is the whole space, but if you remove any vector, the span shrinks.

## Why This Concept Matters

The span is the fundamental building block for describing subspaces. Every subspace can be expressed as the span of some set of vectors. The column space of a matrix $A$ (the span of its columns) is the set of all possible outputs $A\mathbf{x}$. In linear regression, the predicted values $\hat{\mathbf{y}}$ always lie in the column space of the design matrix $X$. The concept of span therefore directly connects linear algebra to data fitting.

Understanding span is essential for:
- Characterizing the solution set of a linear system
- Understanding when a system has a solution ($\mathbf{b}$ must be in the span of the columns)
- Defining rank (the dimension of the column span)
- Building basis expansions for machine learning models
- Grasping the fundamental theorem of linear algebra

## Historical Background

The term "span" emerged from geometric intuition — the idea that vectors "span out" a region of space. Grassmann's *Ausdehnungslehre* (1844) developed the concept of a "linear manifold," which is exactly the span of a set of vectors. The modern notation $\text{span}(S)$ and its systematic study as a subspace was formalized in the early 20th century as vector space theory matured.

The concept of span is intimately tied to solving linear systems. Gauss (1777–1855) developed elimination techniques that implicitly identified whether a vector lies in the span of others, centuries before the formal definition. The connection between span and solvability ($A\mathbf{x} = \mathbf{b}$ has a solution iff $\mathbf{b} \in \text{span}(\text{columns of } A)$) is a cornerstone of linear algebra.

## Real World Examples

1. **Robotics**: A robotic arm with joint angles defines a configuration space. The reachable end-effector positions are the span of the arm's joint movements. If the arm has 3 joints in a plane, the span is the reachable region.

2. **Economics**: In input-output models, the set of all producible goods is the span of the columns of the technology matrix. Each column represents a production process.

3. **Computer graphics**: A 3D scene is rendered by applying transformations to basis vectors. The span of the basis vectors of the camera coordinate system determines what is visible.

4. **Structural engineering**: The set of all possible displacement states of a truss is the span of its mode shapes (eigenvectors of the stiffness matrix).

5. **Audio mixing**: In a multi-track recording, the final mix is a linear combination (span) of individual tracks, weighted by volume sliders (coefficients).

## AI/ML Relevance

**Column space in linear regression**: In ordinary least squares, the model is $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$. The predicted values $\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}} = X(X^T X)^{-1} X^T \mathbf{y}$ lie in $\text{span}(X)$, the column space of the design matrix. The hat matrix $H = X(X^T X)^{-1} X^T$ projects $\mathbf{y}$ onto this span. The residuals $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ are orthogonal to this span.

**Feature spaces spanned by basis expansions**: Polynomial regression uses the span of $\{1, x, x^2, \ldots, x^d\}$. Spline regression uses the span of piecewise polynomial basis functions. The model's expressive power is determined by the span of the chosen feature space. Adding features expands the span, potentially reducing bias but increasing variance.

**Neural network hidden representations**: Each layer of a neural network computes a new representation $\mathbf{h}^{(l)} = \sigma(W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$. The set of all possible representations at layer $l$ (over all inputs) is the span of the columns of the weight matrix $W^{(l)}$, restricted by the activation function.

**Kernel methods**: The kernel $K$ defines a feature map $\phi(\mathbf{x})$. The span of $\{\phi(\mathbf{x}_1), \ldots, \phi(\mathbf{x}_n)\}$ in the RKHS is the space of functions representable as $\sum_i \alpha_i K(\cdot, \mathbf{x}_i)$. The representer theorem guarantees the optimal solution lies in this span.

**Principal component analysis**: PCA projects data onto the span of the top $k$ principal components. This span captures the maximum variance of any $k$-dimensional subspace. The projection is the best rank-$k$ approximation of the data in the least-squares sense.

## Mathematical Explanation

### The Span is a Subspace

**Theorem**: For any set $S \subseteq V$, $\text{span}(S)$ is a subspace of $V$.

**Proof**:

1. **Zero vector**: $0\mathbf{v}_1 + \cdots + 0\mathbf{v}_k = \mathbf{0}$, so $\mathbf{0} \in \text{span}(S)$.

2. **Closure under addition**: If $\mathbf{u} = \sum a_i \mathbf{v}_i$ and $\mathbf{w} = \sum b_i \mathbf{v}_i$ are in $\text{span}(S)$, then $\mathbf{u} + \mathbf{w} = \sum (a_i + b_i) \mathbf{v}_i \in \text{span}(S)$.

3. **Closure under scalar multiplication**: If $\mathbf{u} = \sum a_i \mathbf{v}_i$ and $c \in \mathbb{F}$, then $c\mathbf{u} = \sum (c a_i) \mathbf{v}_i \in \text{span}(S)$.

Thus $\text{span}(S)$ satisfies the subspace test.

### Determining if a Vector is in the Span

To check if $\mathbf{b} \in \text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)$, solve the linear system:

$$a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_k \mathbf{v}_k = \mathbf{b}$$

If the system is consistent, $\mathbf{b}$ is in the span. If inconsistent, $\mathbf{b}$ is not in the span.

In $\mathbb{R}^n$, form the augmented matrix $[\mathbf{v}_1 \; \mathbf{v}_2 \; \cdots \; \mathbf{v}_k \; | \; \mathbf{b}]$ and row-reduce. Consistency indicates $\mathbf{b} \in \text{span}$.

### Spanning a Subspace

A set $S$ is said to **span** a subspace $W$ if $\text{span}(S) = W$. This means:

1. $S \subseteq W$ (every vector in $S$ belongs to $W$).
2. Every vector in $W$ can be expressed as a linear combination of vectors in $S$.

A spanning set can have redundant vectors (vectors that are themselves in the span of the others). Removing redundant vectors yields a basis for the subspace.

### Relation to Basis

A basis is a spanning set that is also linearly independent. This means:

- **Basis = span + independence**: A basis is an **efficient** spanning set with no redundancy.
- Any spanning set can be reduced to a basis by removing vectors that are linear combinations of the others.
- Any linearly independent set can be extended to a basis by adding vectors until it spans.

### Column Space, Row Space, and Nullspace

For an $m \times n$ matrix $A$:

- **Column space**: $\text{col}(A) = \text{span}(\text{columns of } A) \subseteq \mathbb{R}^m$. The set of all possible outputs $A\mathbf{x}$.
- **Row space**: $\text{row}(A) = \text{span}(\text{rows of } A) \subseteq \mathbb{R}^n$. Same as $\text{col}(A^T)$.
- **Nullspace**: $\text{null}(A) = \{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}$. The set of all vectors orthogonal to the rows.
- **Left nullspace**: $\text{null}(A^T) = \{\mathbf{y} \in \mathbb{R}^m \mid A^T \mathbf{y} = \mathbf{0}\}$.

The **rank** of $A$ is $\dim(\text{col}(A)) = \dim(\text{row}(A))$. The rank-nullity theorem states:
$$\text{rank}(A) + \dim(\text{null}(A)) = n$$

## Formula(s)

**Span definition**:
$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\{ \sum_{i=1}^{k} a_i \mathbf{v}_i \;\middle|\; a_i \in \mathbb{F} \right\}$$

**Column space**:
$$\text{col}(A) = \{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n\} = \text{span}(\mathbf{a}_1, \ldots, \mathbf{a}_n)$$

where $\mathbf{a}_i$ are columns of $A$.

**Projection onto span** (for orthonormal basis $\{\mathbf{q}_1, \ldots, \mathbf{q}_r\}$):
$$\text{proj}_{\text{span}}(\mathbf{b}) = \sum_{i=1}^{r} (\mathbf{q}_i \cdot \mathbf{b}) \mathbf{q}_i$$

## Properties

1. **Span is a subspace**: $\text{span}(S)$ is always a subspace of $V$, for any $S \subseteq V$.
2. **Smallest containing subspace**: $\text{span}(S)$ is the smallest subspace of $V$ containing $S$.
3. **Monotonicity**: If $S \subseteq T$, then $\text{span}(S) \subseteq \text{span}(T)$.
4. **Idempotence**: $\text{span}(\text{span}(S)) = \text{span}(S)$.
5. **Span of a basis**: If $\mathcal{B}$ is a basis for $V$, then $\text{span}(\mathcal{B}) = V$.
6. **Redundancy**: If $\mathbf{v}_i \in \text{span}(S \setminus \{\mathbf{v}_i\})$, then $\text{span}(S) = \text{span}(S \setminus \{\mathbf{v}_i\})$.
7. **Span of the empty set**: $\text{span}(\emptyset) = \{\mathbf{0}\}$.
8. **Intersection property**: $\text{span}(S) \cap \text{span}(T)$ is a subspace (but not necessarily equal to $\text{span}(S \cap T)$).
9. **Sum of spans**: $\text{span}(S) + \text{span}(T) = \text{span}(S \cup T)$.
10. **Dimension bound**: $\dim(\text{span}(S)) \leq |S|$, with equality iff $S$ is linearly independent.

## Step-by-Step Worked Examples

### Example 1: Determining if a Vector is in the Span

**Problem**: Determine whether $\mathbf{b} = (1, 2, 3)$ is in the span of $\mathbf{v}_1 = (1, 0, 1)$ and $\mathbf{v}_2 = (0, 1, 1)$.

**Solution**:

We need to check if there exist $a_1, a_2$ such that $a_1(1, 0, 1) + a_2(0, 1, 1) = (1, 2, 3)$.

Set up the system:
$$a_1 + 0a_2 = 1$$
$$0a_1 + a_2 = 2$$
$$a_1 + a_2 = 3$$

From the first equation: $a_1 = 1$.
From the second: $a_2 = 2$.
Check the third: $1 + 2 = 3$. ✓

Thus $\mathbf{b} = 1 \cdot \mathbf{v}_1 + 2 \cdot \mathbf{v}_2$, so $\mathbf{b} \in \text{span}(\mathbf{v}_1, \mathbf{v}_2)$.

### Example 2: Determining Span of Three Vectors in $\mathbb{R}^3$

**Problem**: Find the span of $\mathbf{v}_1 = (1, 1, 0)$, $\mathbf{v}_2 = (0, 1, 1)$, $\mathbf{v}_3 = (1, 2, 1)$.

**Solution**:

Form the matrix with these vectors as columns and row-reduce:
$$A = \begin{pmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \\ 0 & 1 & 1 \end{pmatrix}$$

Row reduce:
$$R_2 \leftarrow R_2 - R_1: \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$
$$R_3 \leftarrow R_3 - R_2: \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

The pivot columns are columns 1 and 2. Column 3 is $\mathbf{v}_3 = \mathbf{v}_1 + \mathbf{v}_2$, so it is redundant.

Thus $\text{span}(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3) = \text{span}(\mathbf{v}_1, \mathbf{v}_2)$.

Check if $\mathbf{v}_1$ and $\mathbf{v}_2$ are independent:
$a(1, 1, 0) + b(0, 1, 1) = (0, 0, 0)$ gives $a = 0$, $a + b = 0 \implies b = 0$, $b = 0$. So they are independent.

Thus the span is a 2-dimensional subspace (a plane) of $\mathbb{R}^3$. The plane consists of all vectors:
$$\mathbf{v} = a(1, 1, 0) + b(0, 1, 1) = (a, a + b, b)$$

The equation of this plane: from $x = a$, $z = b$, $y = a + b = x + z$, so the plane is $y = x + z$.

### Example 3: Column Space and Consistency

**Problem**: Let $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}$ and $\mathbf{b} = \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}$. Determine whether $A\mathbf{x} = \mathbf{b}$ is consistent.

**Solution**:

The system is consistent iff $\mathbf{b} \in \text{col}(A) = \text{span}(\mathbf{a}_1, \mathbf{a}_2)$, where $\mathbf{a}_1 = (1, 2, 3)^T$ and $\mathbf{a}_2 = (2, 4, 6)^T$.

Note that $\mathbf{a}_2 = 2\mathbf{a}_1$, so $\text{col}(A) = \text{span}(\mathbf{a}_1) = \{t(1, 2, 3)^T \mid t \in \mathbb{R}\}$.

Thus $\mathbf{b}$ is in the column space iff $\mathbf{b}$ is a scalar multiple of $(1, 2, 3)^T$.

Check: $(1, 2, 4)^T \stackrel{?}{=} t(1, 2, 3)^T$ gives $t = 1$, $t = 1$, $t = 4/3$, which is inconsistent.

Therefore $\mathbf{b} \notin \text{col}(A)$, and $A\mathbf{x} = \mathbf{b}$ has no solution.

**Alternative approach**: Form augmented matrix and row-reduce:
$$\begin{pmatrix} 1 & 2 & | & 1 \\ 2 & 4 & | & 2 \\ 3 & 6 & | & 4 \end{pmatrix}$$

Row reduce:
$$R_2 \leftarrow R_2 - 2R_1: \begin{pmatrix} 1 & 2 & | & 1 \\ 0 & 0 & | & 0 \\ 3 & 6 & | & 4 \end{pmatrix}$$
$$R_3 \leftarrow R_3 - 3R_1: \begin{pmatrix} 1 & 2 & | & 1 \\ 0 & 0 & | & 0 \\ 0 & 0 & | & 1 \end{pmatrix}$$

The last row gives $0 = 1$, which is inconsistent. So no solution exists.

### Example 4: Finding a Spanning Set for a Subspace

**Problem**: Find a spanning set for the subspace $W = \{(x, y, z, w) \in \mathbb{R}^4 \mid x + 2y - z + w = 0\}$.

**Solution**:

Express one variable in terms of the others. From $x + 2y - z + w = 0$, we have $x = -2y + z - w$.

Any vector in $W$ can be written as:
$$(x, y, z, w) = (-2y + z - w, y, z, w)$$

Write as a linear combination:
$$= y(-2, 1, 0, 0) + z(1, 0, 1, 0) + w(-1, 0, 0, 1)$$

Thus $W = \text{span}\{(-2, 1, 0, 0), (1, 0, 1, 0), (-1, 0, 0, 1)\}$.

Check independence: $a(-2, 1, 0, 0) + b(1, 0, 1, 0) + c(-1, 0, 0, 1) = (0, 0, 0, 0)$ gives:
$-2a + b - c = 0$, $a = 0$, $b = 0$, $c = 0 \implies a = b = c = 0$.

So this set is a basis for $W$, and $\dim(W) = 3$.

### Example 5: Projection onto a Span

**Problem**: Find the projection of $\mathbf{b} = (4, 5, 6)$ onto $\text{span}(\mathbf{v}_1, \mathbf{v}_2)$ where $\mathbf{v}_1 = (1, 0, 0)$, $\mathbf{v}_2 = (0, 1, 1)$. Assume the standard inner product.

**Solution**:

Note that $\mathbf{v}_1$ and $\mathbf{v}_2$ are independent but not orthogonal ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$, so actually they are orthogonal!).

Since they are orthogonal, we can use the formula for projection onto an orthogonal basis:
$$\text{proj}(\mathbf{b}) = \frac{\mathbf{b} \cdot \mathbf{v}_1}{\mathbf{v}_1 \cdot \mathbf{v}_1} \mathbf{v}_1 + \frac{\mathbf{b} \cdot \mathbf{v}_2}{\mathbf{v}_2 \cdot \mathbf{v}_2} \mathbf{v}_2$$

Compute:
$$\mathbf{b} \cdot \mathbf{v}_1 = 4(1) + 5(0) + 6(0) = 4, \quad \mathbf{v}_1 \cdot \mathbf{v}_1 = 1$$
$$\mathbf{b} \cdot \mathbf{v}_2 = 4(0) + 5(1) + 6(1) = 11, \quad \mathbf{v}_2 \cdot \mathbf{v}_2 = 1^2 + 1^2 = 2$$

Thus:
$$\text{proj}(\mathbf{b}) = \frac{4}{1}(1, 0, 0) + \frac{11}{2}(0, 1, 1) = (4, 0, 0) + (0, 5.5, 5.5) = (4, 5.5, 5.5)$$

**Verification**: The residual $\mathbf{b} - \text{proj}(\mathbf{b}) = (0, -0.5, 0.5)$ should be orthogonal to the span:
- Dot with $\mathbf{v}_1$: $(0, -0.5, 0.5) \cdot (1, 0, 0) = 0$ ✓
- Dot with $\mathbf{v}_2$: $(0, -0.5, 0.5) \cdot (0, 1, 1) = 0 - 0.5 + 0.5 = 0$ ✓

### Example 6: Span and Linear Regression

**Problem**: In simple linear regression, the design matrix is $X = \begin{pmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{pmatrix}$. The column space is $\text{span}(\mathbf{1}, \mathbf{x})$ where $\mathbf{1} = (1, 1, \ldots, 1)^T$ and $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$. Show that the predicted values $\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}}$ lie in this span.

**Solution**:

The OLS estimate is $\hat{\boldsymbol{\beta}} = (\beta_0, \beta_1)^T = (X^T X)^{-1} X^T \mathbf{y}$.

The predicted values are:
$$\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}} = \beta_0 \mathbf{1} + \beta_1 \mathbf{x}$$

This is explicitly a linear combination of $\mathbf{1}$ and $\mathbf{x}$, so $\hat{\mathbf{y}} \in \text{span}(\mathbf{1}, \mathbf{x}) = \text{col}(X)$.

The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ satisfies $X^T \mathbf{e} = \mathbf{0}$, meaning $\mathbf{e}$ is orthogonal to $\text{col}(X)$.

## Visual Interpretation

In $\mathbb{R}^2$, the span of a single non-zero vector is a line through the origin. The span of two non-collinear vectors is the entire plane $\mathbb{R}^2$.

In $\mathbb{R}^3$, the possibilities are richer:
- Span of one non-zero vector: a line through the origin
- Span of two non-parallel, non-zero vectors: a plane through the origin
- Span of three linearly independent vectors: all of $\mathbb{R}^3$
- Span of three vectors where one is redundant: a plane (if two are independent) or a line (if all are collinear)

The span is always a **flat through the origin**. It is never a curved surface, never a line that does not pass through the origin, and never a bounded region. As a subspace, it is always infinite in extent.

A useful mental model: the span is the set of all points that can be reached from the origin by walking along the given directions. If you have vectors $\mathbf{v}_1$ and $\mathbf{v}_2$, you can walk any distance along $\mathbf{v}_1$, then any distance along $\mathbf{v}_2$, and those two steps uniquely determine your position.

## Common Mistakes

1. **Confusing span with the set of vectors itself**: The span is the set of all linear combinations, not just the original vectors. For example, $\text{span}\{(1, 0)\}$ includes $(2, 0)$, $(3, 0)$, $(-5, 0)$, etc., not just $(1, 0)$.

2. **Assuming more vectors always means larger span**: Adding a vector that is already in the span does not increase the span. The span is determined by the number of independent vectors, not the total number.

3. **Thinking the span of $k$ vectors in $\mathbb{R}^n$ always has dimension $k$**: The dimension of the span equals the number of linearly independent vectors, which may be less than $k$.

4. **Believing the span must be the entire ambient space**: $\text{span}\{(1, 0, 0), (0, 1, 0)\}$ is a plane (the $xy$-plane), not all of $\mathbb{R}^3$. A spanning set may not span the whole space.

5. **Forgetting to check consistency when testing if a vector is in a span**: Set up the linear system and row-reduce. A quick glance is not sufficient.

6. **Confusing column space with row space**: The column space of $A$ is the span of columns (in $\mathbb{R}^m$). The row space is the span of rows (in $\mathbb{R}^n$). They are different spaces (though they have the same dimension).

7. **Assuming the span is always finite-dimensional**: The span of an infinite set of vectors can be infinite-dimensional. For example, $\text{span}\{1, x, x^2, x^3, \ldots\}$ in the space of all polynomials is infinite-dimensional.

8. **Not recognizing that span depends on the field**: $\text{span}_{\mathbb{R}}\{1, \sqrt{2}\}$ in $\mathbb{R}$ (as a vector space over $\mathbb{R}$) is $\mathbb{R}$, but $\text{span}_{\mathbb{Q}}\{1, \sqrt{2}\}$ in $\mathbb{R}$ (as a vector space over $\mathbb{Q}$) is $\{a + b\sqrt{2} \mid a, b \in \mathbb{Q}\}$, a 2-dimensional space.

9. **Thinking a non-zero vector does not span a subspace**: A single non-zero vector spans a 1-dimensional subspace (a line). The span is always a subspace, even for a single vector.

10. **Confusing span with linear combination**: A linear combination is a single expression; the span is the infinite set of all possible linear combinations. The distinction between an individual element and the entire set is crucial.

## Interview Questions

### Beginner

**Q1**: What is the span of a set of vectors?

**A**: The span is the set of all finite linear combinations of the vectors. It forms a subspace of the ambient vector space.

**Q2**: What is the span of $\{(1, 0), (0, 1)\}$ in $\mathbb{R}^2$?

**A**: All of $\mathbb{R}^2$, since any $(x, y) = x(1, 0) + y(0, 1)$.

**Q3**: What is the span of a single non-zero vector $\mathbf{v}$?

**A**: A line through the origin: $\text{span}(\mathbf{v}) = \{a\mathbf{v} \mid a \in \mathbb{R}\}$.

**Q4**: Is the span of any set of vectors always a subspace?

**A**: Yes. The span always contains the zero vector (all coefficients zero), is closed under addition (sum of linear combinations is a linear combination), and closed under scalar multiplication (scaling a linear combination is a linear combination).

**Q5**: Can the span of 3 vectors in $\mathbb{R}^3$ be a line?

**A**: Yes, if all three vectors are collinear (scalar multiples of each other). For example, $\{(1, 0, 0), (2, 0, 0), (-3, 0, 0)\}$ spans the $x$-axis, a line.

### Intermediate

**Q1**: What is the column space of a matrix? Why is it important?

**A**: The column space of $A \in \mathbb{R}^{m \times n}$ is $\text{col}(A) = \text{span}(\text{columns of }A) \subseteq \mathbb{R}^m$. It is the set of all vectors $\mathbf{b}$ for which $A\mathbf{x} = \mathbf{b}$ is solvable. It determines the range of the linear transformation defined by $A$.

**Q2**: Find a spanning set for the nullspace of $A = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$.

**A**: Solve $A\mathbf{x} = \mathbf{0}$:
$$x_1 + x_2 + x_3 = 0$$
$$x_2 + x_3 = 0 \implies x_2 = -x_3$$
Then $x_1 = -x_2 - x_3 = -(-x_3) - x_3 = x_3 - x_3 = 0$.
So $\mathbf{x} = (0, -x_3, x_3) = x_3(0, -1, 1)$. Spanning set: $\{(0, -1, 1)\}$, nullspace is 1-dimensional.

**Q3**: Prove that $\text{span}(S \cup T) = \text{span}(S) + \text{span}(T)$.

**A**: ($\subseteq$) Any $\mathbf{v} \in \text{span}(S \cup T)$ is a linear combination of vectors from $S \cup T$. Group the $S$ terms and $T$ terms to write $\mathbf{v} = \mathbf{s} + \mathbf{t}$ where $\mathbf{s} \in \text{span}(S)$ and $\mathbf{t} \in \text{span}(T)$. So $\mathbf{v} \in \text{span}(S) + \text{span}(T)$.

($\supseteq$) Any $\mathbf{v} \in \text{span}(S) + \text{span}(T)$ is $\mathbf{v} = \mathbf{s} + \mathbf{t}$ with $\mathbf{s} \in \text{span}(S)$, $\mathbf{t} \in \text{span}(T)$. Each of $\mathbf{s}, \mathbf{t}$ is a linear combination of vectors in $S$ and $T$, so $\mathbf{v}$ is a linear combination of vectors in $S \cup T$.

**Q4**: In linear regression, why do the fitted values lie in the span of the design matrix columns?

**A**: $\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}} = \sum_{j=1}^p \hat{\beta}_j \mathbf{x}_j$ is explicitly a linear combination of the columns of $X$. Hence $\hat{\mathbf{y}} \in \text{col}(X)$. Equivalently, $\hat{\mathbf{y}}$ is the orthogonal projection of $\mathbf{y}$ onto $\text{col}(X)$.

**Q5**: What is the relationship between the rank of a matrix and the dimension of its column space?

**A**: The rank of $A$ is exactly $\dim(\text{col}(A))$, the number of linearly independent columns. It is also $\dim(\text{row}(A))$.

### Advanced

**Q1**: State and prove the rank-nullity theorem using the concept of span.

**A**: For $A \in \mathbb{R}^{m \times n}$, let $T(\mathbf{x}) = A\mathbf{x}$. Let $\{\mathbf{v}_1, \ldots, \mathbf{v}_r\}$ be a basis for $\text{col}(A)$ where $r = \text{rank}(A)$. Let $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ be a basis for $\text{null}(A)$. Extend the nullspace basis to a basis for $\mathbb{R}^n$: $\{\mathbf{u}_1, \ldots, \mathbf{u}_k, \mathbf{w}_1, \ldots, \mathbf{w}_{n-k}\}$. Then $\{T(\mathbf{w}_1), \ldots, T(\mathbf{w}_{n-k})\}$ spans $\text{col}(A)$ and is independent, so $n - k = r$, i.e., $n = \dim(\text{null}(A)) + \text{rank}(A)$.

**Q2**: Let $S = \{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ be a set of vectors in $V$. Prove that $\text{span}(S)$ is the intersection of all subspaces of $V$ that contain $S$.

**A**: Let $\mathcal{F} = \{W \subseteq V \mid W \text{ is a subspace and } S \subseteq W\}$. Let $U = \bigcap_{W \in \mathcal{F}} W$. We show $U = \text{span}(S)$.

($\text{span}(S) \subseteq U$): Since $\text{span}(S)$ is itself a subspace containing $S$, it is in $\mathcal{F}$, so $U \subseteq \text{span}(S)$. Wait — $U$ is the intersection, so it is contained in each $W$, including $\text{span}(S)$. So $U \subseteq \text{span}(S)$.

Actually let's be careful. $U = \bigcap_{W \in \mathcal{F}} W$. Since $\text{span}(S) \in \mathcal{F}$, $U \subseteq \text{span}(S)$. Conversely, every $W \in \mathcal{F}$ contains $S$, hence contains every linear combination of vectors in $S$, so $\text{span}(S) \subseteq W$ for all $W \in \mathcal{F}$. Thus $\text{span}(S) \subseteq \bigcap_{W \in \mathcal{F}} W = U$. Therefore $U = \text{span}(S)$.

This justifies calling $\text{span}(S)$ the "smallest subspace containing $S$."

**Q3**: Explain how the concept of span relates to basis expansions in machine learning. Give a concrete example where a linear model in an expanded feature space can fit non-linear data.

**A**: Linear models in the original features can only fit linear relationships. By expanding the feature space (increasing the span of the feature set), we can fit non-linear relationships while remaining linear in the parameters.

Example: Consider data $(x_i, y_i)$ where $y_i = \sin(x_i)$. A simple linear model $y = \beta_0 + \beta_1 x$ fits poorly. Expand features to $\{1, x, x^2, x^3\}$ (polynomial basis). The model $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$ is linear in $\beta$ but can approximate $\sin(x)$ near $x = 0$. The feature vector is $\phi(x) = (1, x, x^2, x^3)$, and the prediction $\hat{y} = \boldsymbol{\beta}^T \phi(x)$ lies in $\text{span}\{1, x, x^2, x^3\}$, which is a 4-dimensional function space that includes non-linear functions of $x$.

More generally, a basis expansion replaces $\mathbf{x} \in \mathbb{R}^d$ with $\phi(\mathbf{x}) \in \mathbb{R}^p$ where $p \gg d$. The columns of the design matrix are the $p$ basis functions evaluated at the $n$ data points. The span of these $p$ columns determines what functions the model can represent. With enough basis functions (e.g., splines, Fourier, RBFs), the model can approximate any continuous function arbitrarily well.

## Practice Problems

### Easy — 5 Questions

**E1**: Is $\mathbf{b} = (5, 7)$ in $\text{span}\{(1, 2), (3, 4)\}$?

**E2**: What is the span of $\{(0, 0)\}$ in $\mathbb{R}^2$?

**E3**: Find a vector in $\mathbb{R}^3$ that is NOT in $\text{span}\{(1, 1, 0), (0, 0, 1)\}$.

**E4**: True or False: $\text{span}\{(1, 2), (2, 4)\}$ is a line.

**E5**: Does $(1, 0, 0)$ span all of $\mathbb{R}^3$? Explain.

### Medium — 5 Questions

**M1**: Find the equation of the plane that is the span of $\{(1, 2, 0), (0, 1, 1)\}$.

**M2**: Determine if $\mathbf{b} = (1, 2, 3, 4)$ is in $\text{span}\{(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 1, 1)\}$.

**M3**: Find a spanning set for $W = \{(x, y, z, w) \in \mathbb{R}^4 \mid x + y + z + w = 0 \text{ and } x - y = 0\}$.

**M4**: Let $A = \begin{pmatrix} 1 & 0 & 2 \\ 2 & 1 & 5 \\ 0 & 1 & 1 \end{pmatrix}$. Find a basis for $\text{col}(A)$ and determine its dimension.

**M5**: Show that $\text{span}\{(1, 2, 3), (4, 5, 6), (7, 8, 9)\}$ is a 2-dimensional subspace of $\mathbb{R}^3$.

### Hard — 3 Questions

**H1**: Let $V$ be the vector space of all $2 \times 2$ matrices. Find the span of $\left\{ \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} \right\}$ and find its dimension.

**H2**: Prove that $\text{span}(U \cup W) = U + W$ for any two subspaces $U, W$ of $V$.

**H3**: Let $\mathbf{v}_1, \ldots, \mathbf{v}_k$ be vectors in $\mathbb{R}^n$. Prove that $\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \text{row}(A)$ where $A$ is the $k \times n$ matrix with rows $\mathbf{v}_1^T, \ldots, \mathbf{v}_k^T$. Show that $\text{dim}(\text{span}) = \text{rank}(A)$.

## Solutions

### Easy Solutions

**E1**: Solve $a(1, 2) + b(3, 4) = (5, 7)$:
$$a + 3b = 5$$
$$2a + 4b = 7$$
From first: $a = 5 - 3b$. Substitute: $2(5 - 3b) + 4b = 10 - 6b + 4b = 10 - 2b = 7 \implies 2b = 3 \implies b = 1.5$, then $a = 5 - 4.5 = 0.5$. System is consistent, so yes.

**E2**: $\text{span}\{(0, 0)\} = \{\mathbf{0}\}$, the zero subspace.

**E3**: The span of $\{(1, 1, 0), (0, 0, 1)\}$ consists of $(a, a, b)$ for $a, b \in \mathbb{R}$. Any vector where the first two components differ is not in the span. For example, $(1, 0, 0)$ is not in the span.

**E4**: True. $(2, 4) = 2(1, 2)$, so the vectors are collinear. The span is $\{t(1, 2) \mid t \in \mathbb{R}\}$, a line through the origin.

**E5**: No. $\text{span}\{(1, 0, 0)\}$ is the $x$-axis. Vectors with non-zero $y$ or $z$ components (e.g., $(0, 1, 0)$) are not in the span.

### Medium Solutions

**M1**: A general vector in the span is $a(1, 2, 0) + b(0, 1, 1) = (a, 2a + b, b)$. So $x = a$, $z = b$, $y = 2a + b = 2x + z$. The plane equation is $y = 2x + z$ or $2x - y + z = 0$.

**M2**: Solve $a(1, 0, 1, 0) + b(0, 1, 0, 1) + c(1, 1, 1, 1) = (1, 2, 3, 4)$:
Component 1: $a + c = 1$
Component 2: $b + c = 2$
Component 3: $a + c = 3$
Component 4: $b + c = 4$

From components 1 and 3: $a + c = 1$ and $a + c = 3$, contradiction. So no solution, $\mathbf{b}$ is not in the span.

**M3**: Conditions: $x + y + z + w = 0$ and $x - y = 0 \implies x = y$.
Then $x + x + z + w = 0 \implies 2x + z + w = 0 \implies w = -2x - z$.
Vector: $(x, x, z, -2x - z) = x(1, 1, 0, -2) + z(0, 0, 1, -1)$.
Spanning set: $\{(1, 1, 0, -2), (0, 0, 1, -1)\}$.

**M4**: Row-reduce $A$:
$$\begin{pmatrix} 1 & 0 & 2 \\ 2 & 1 & 5 \\ 0 & 1 & 1 \end{pmatrix}$$
$$R_2 \leftarrow R_2 - 2R_1: \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$
$$R_3 \leftarrow R_3 - R_2: \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

Pivot columns: 1 and 2. Basis for $\text{col}(A)$: $\{(1, 2, 0)^T, (0, 1, 1)^T\}$, dimension 2.

**M5**: Form matrix with vectors as rows and row-reduce:
$$\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$
$$R_2 \leftarrow R_2 - 4R_1: \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 7 & 8 & 9 \end{pmatrix}$$
$$R_3 \leftarrow R_3 - 7R_1: \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & -6 & -12 \end{pmatrix}$$
$$R_3 \leftarrow R_3 - 2R_2: \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & 0 & 0 \end{pmatrix}$$

Rank = 2, so the span is 2-dimensional. Indeed, $(7, 8, 9) = 2(4, 5, 6) - (1, 2, 3)$, so the third vector is redundant.

### Hard Solutions

**H1**: Let $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $S = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $T = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$. The span consists of all matrices:
$$a\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} + b\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} + c\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} = \begin{pmatrix} a & b + c \\ b - c & a \end{pmatrix}$$

This is the space of all $2 \times 2$ matrices of the form $\begin{pmatrix} a & d \\ e & a \end{pmatrix}$ where $d = b + c$ and $e = b - c$ are unrestricted ($b, c$ can produce any $d, e$). The dimension is 3. Basis: $\{I, S, T\}$, and they are independent since $aI + bS + cT = \mathbf{0}$ implies $a = b = c = 0$.

**H2**: $U + W = \{\mathbf{u} + \mathbf{w} \mid \mathbf{u} \in U, \mathbf{w} \in W\}$.

($\subseteq$) $U \cup W \subseteq U + W$ (since $\mathbf{u} \in U$ gives $\mathbf{u} = \mathbf{u} + \mathbf{0} \in U + W$, similarly for $W$). The span of $U \cup W$ is the smallest subspace containing $U \cup W$, and $U + W$ is a subspace containing $U \cup W$, so $\text{span}(U \cup W) \subseteq U + W$.

($\supseteq$) Any $\mathbf{v} \in U + W$ is $\mathbf{v} = \mathbf{u} + \mathbf{w}$ with $\mathbf{u} \in U$, $\mathbf{w} \in W$. Since $\mathbf{u}, \mathbf{w} \in U \cup W$, their sum is a linear combination of vectors in $U \cup W$, so $\mathbf{v} \in \text{span}(U \cup W)$. Thus $U + W \subseteq \text{span}(U \cup W)$.

Therefore $\text{span}(U \cup W) = U + W$.

**H3**: Let $A$ be the $k \times n$ matrix with rows $\mathbf{v}_1^T, \ldots, \mathbf{v}_k^T$. The row space $\text{row}(A)$ is the span of the rows: $\text{span}\{\mathbf{v}_1^T, \ldots, \mathbf{v}_k^T\}$. Identifying row vectors with column vectors, $\text{row}(A)$ is exactly $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$.

The rank of $A$ is the dimension of its row space, which equals the dimension of the span. Row reduction does not change the row space. After row reduction to RREF, the non-zero rows form a basis for the row space, and the number of non-zero rows equals the rank. Hence $\dim(\text{span}) = \text{rank}(A)$.

## Related Concepts

- **Vector space (MATH-031)**: The ambient space containing spans.
- **Basis (MATH-032)**: A minimal spanning set (spanning + independence).
- **Linear combination**: The building block of span — every span element is a linear combination.
- **Linear independence**: Vectors are independent if none is in the span of the others.
- **Column space**: The span of the columns of a matrix; key to solving $A\mathbf{x} = \mathbf{b}$.
- **Row space**: The span of the rows; unchanged by row operations.
- **Nullspace**: The set of $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$; orthogonal to row space.
- **Rank**: The dimension of the column space (or row space).
- **Projection**: The orthogonal projection onto a subspace (span) minimizes distance.
- **Fundamental theorem of linear algebra**: Relates the four fundamental subspaces (column, row, null, left null) of a matrix.

## Next Concepts

- Linear transformations and their matrix representations
- Eigenvalues and eigenvectors
- Orthogonality and Gram-Schmidt
- Least squares and orthogonal projection
- Singular value decomposition
- Principal component analysis

## Summary

The span of a set of vectors is the set of all possible linear combinations of those vectors. It is always a subspace — the smallest subspace containing the original vectors. The concept of span is fundamental to understanding the column space of a matrix (which determines system solvability), the expressive power of feature expansions in machine learning, and the relationship between vectors and the subspaces they generate. A basis is a spanning set with no redundancy (linear independence). The dimension of a span is the number of linearly independent vectors in the generating set.

## Key Takeaways

1. The span of a set $S$ is the set of all linear combinations of vectors in $S$.
2. The span of any set is always a subspace of the ambient vector space.
3. A vector $\mathbf{b}$ is in $\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)$ iff the linear system $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k = \mathbf{b}$ has a solution.
4. Adding a redundant vector (already in the span) does not expand the span.
5. The column space of a matrix $A$ is the span of its columns; $A\mathbf{x} = \mathbf{b}$ is solvable iff $\mathbf{b} \in \text{col}(A)$.
6. A basis is a spanning set that is also linearly independent — minimal and efficient.
7. The dimension of a span equals the rank of the matrix formed by the vectors.
8. In linear regression, $\hat{\mathbf{y}}$ lies in $\text{col}(X)$ — the span of the design matrix columns.
9. Basis expansions (polynomial, Fourier, RBF) increase the span of features, enabling non-linear fitting with linear models.
10. The span is the "smallest subspace containing the set" — it represents all vectors reachable via the given directions.
