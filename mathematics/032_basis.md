# Concept: Basis

## Concept ID

MATH-032

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

1. Define a basis as a linearly independent spanning set of a vector space
2. Determine whether a given set of vectors forms a basis for a vector space
3. Compute the coordinates of a vector relative to a given basis
4. Apply the concept of dimension as the number of basis vectors
5. Perform change-of-basis computations using transition matrices
6. Connect basis concepts to feature engineering and dimensionality reduction in machine learning

## Prerequisites

- Vector spaces and subspaces (MATH-031)
- Linear combinations and the span of a set of vectors
- Linear independence: a set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is linearly independent if $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k = \mathbf{0}$ implies all $a_i = 0$
- Systems of linear equations and Gaussian elimination
- Matrix notation and multiplication

## Definition

Let $V$ be a vector space over a field $\mathbb{F}$. A **basis** of $V$ is a set of vectors $\mathcal{B} = \{\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n\} \subseteq V$ such that:

1. **Spanning property**: $\text{span}(\mathcal{B}) = V$. Every vector in $V$ can be written as a linear combination of the basis vectors.
2. **Linear independence**: $\mathcal{B}$ is linearly independent. No basis vector can be expressed as a linear combination of the others.

Equivalently, a basis is a **minimal spanning set** (a spanning set with no redundant vectors) or a **maximal linearly independent set** (a linearly independent set that cannot be enlarged while preserving independence).

Every vector $\mathbf{v} \in V$ can be written **uniquely** as a linear combination of basis vectors:
$$\mathbf{v} = c_1 \mathbf{b}_1 + c_2 \mathbf{b}_2 + \cdots + c_n \mathbf{b}_n$$

The scalars $(c_1, c_2, \ldots, c_n)$ are called the **coordinates** of $\mathbf{v}$ relative to $\mathcal{B}$, denoted $[\mathbf{v}]_{\mathcal{B}}$.

The **dimension** of $V$, written $\dim(V)$, is the number of vectors in any basis for $V$. All bases for a given vector space have the same cardinality.

## Intuition

Think of a basis as the minimal set of "building blocks" from which every element of the vector space can be assembled. In $\mathbb{R}^3$, the standard basis $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ corresponds to the three perpendicular axes: $\mathbf{e}_1 = (1, 0, 0)$ points along the $x$-axis, $\mathbf{e}_2 = (0, 1, 0)$ along the $y$-axis, $\mathbf{e}_3 = (0, 0, 1)$ along the $z$-axis. Every point in space is a unique combination of these three directions.

Choosing a basis is like choosing a coordinate system. Different bases give different coordinates for the same vector, just as the same location can be described using Cartesian or polar coordinates. The key property is that the **representation** changes but the **vector itself** does not. This is analogous to describing an object in different languages — the object remains, but the words (coordinates) differ.

A basis must be both **sufficient** (spanning: you can reach every point) and **efficient** (independent: no redundancy). If a set is not spanning, some vectors are unreachable. If it is not independent, some vectors have multiple representations.

## Why This Concept Matters

Bases are the bridge between abstract vector spaces and concrete computation. Without a basis, a vector is an abstract element of an abstract set. With a basis, every vector becomes an $n$-tuple of coordinates, and every linear transformation becomes a matrix. This computational representation is what makes linear algebra practically useful in science and engineering.

In machine learning, choosing the right basis is equivalent to choosing the right representation for data. The standard basis corresponds to raw features. Principal component analysis (PCA) finds a new basis (the eigenbasis of the covariance matrix) that better captures variance. Basis functions (polynomials, radial basis functions, Fourier modes) underpin regression, signal processing, and kernel methods.

## Historical Background

The concept of a basis emerged from the work of several mathematicians. Joseph-Louis Lagrange (1736–1813) used coordinate systems that implicitly relied on basis vectors. Augustin-Louis Cauchy and Carl Friedrich Gauss developed the theory of determinants, which are intimately connected to linear independence and basis detection. 

The term "basis" was introduced by Hermann Grassmann in his 1844 *Ausdehnungslehre*, though his presentation was so abstract that it was not widely appreciated during his lifetime. Giuseppe Peano gave the first rigorous definition of a basis within the axiomatic framework of vector spaces in 1888. 

The modern theory of bases, including the notion that every vector space (even infinite-dimensional) has a basis, relies on the Axiom of Choice (specifically Zorn's Lemma) for infinite-dimensional spaces, a fact established by mathematicians in the early 20th century.

## Real World Examples

1. **Cartesian coordinate system**: The unit vectors $\hat{i}, \hat{j}, \hat{k}$ form a basis for physical three-dimensional space. Any position or force vector is a combination of these three directions.

2. **Color representation**: The RGB color space uses three basis colors (red, green, blue). Any color is a linear combination of these primaries. CMYK printing uses a different basis (cyan, magenta, yellow, black).

3. **Fourier series**: The functions $\{1, \sin(nx), \cos(nx)\}$ form a basis for the space of periodic functions. Any periodic signal can be decomposed into its Fourier coefficients in this basis.

4. **Audio compression**: MP3 encoding uses a basis of frequency components (via the modified discrete cosine transform). The signal is represented in this basis, and less important coefficients are discarded.

5. **Genetics**: In genomics, a small set of "eigengenes" (derived from PCA) can form a basis that captures the most significant variation in gene expression data across thousands of genes.

## AI/ML Relevance

**Basis functions in polynomial regression**: In polynomial regression of degree $d$, the model is
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d + \varepsilon$$
The functions $\{1, x, x^2, \ldots, x^d\}$ form a basis for the space of degree-$d$ polynomials. The feature matrix $X$ has columns corresponding to these basis functions evaluated at each data point.

**Radial basis functions (RBFs)**: RBF networks use basis functions $\phi(\|\mathbf{x} - \mathbf{c}_i\|)$ centered at $k$ points $\mathbf{c}_i$. Common choices include the Gaussian $\phi(r) = \exp(-\gamma r^2)$ and the multiquadric. The model is a linear combination of these basis functions:
$$f(\mathbf{x}) = \sum_{i=1}^{k} w_i \phi(\|\mathbf{x} - \mathbf{c}_i\|)$$
The weights $w_i$ are learned by linear regression. The basis functions transform the input into a higher-dimensional feature space where the target is linearly representable.

**Principal component analysis (PCA)**: PCA finds the **principal basis** — an orthonormal basis ordered by variance. The first basis vector (first principal component) points in the direction of maximum variance. Projecting data onto the first $k$ components gives a low-dimensional representation that preserves as much variance as possible. This is a change of basis from the standard basis to the eigenbasis of the covariance matrix.

**Fourier basis in signal processing**: The Fourier basis $\{e^{i n \omega t}\}$ (or equivalently $\{\sin(n\omega t), \cos(n\omega t)\}$) decomposes signals into frequency components. In convolutional neural networks, Fourier techniques are used for efficient convolution via the convolution theorem. In time series forecasting, seasonal patterns are captured using Fourier basis features.

**Neural network feature spaces**: Each hidden layer of a neural network produces a new representation of the input — effectively a learned basis in which the task becomes easier. The final layer is a linear combination of these learned basis features.

## Mathematical Explanation

### Determining Whether a Set is a Basis

To check if $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is a basis for $V$:

1. **Check spanning**: Show that every vector in $V$ is a linear combination of $\mathcal{B}$. For $\mathbb{R}^n$, this reduces to verifying that the matrix with columns $\mathbf{v}_i$ has rank $n$.
2. **Check independence**: Show that $a_1\mathbf{v}_1 + \cdots + a_k\mathbf{v}_k = \mathbf{0}$ implies $a_1 = \cdots = a_k = 0$. This is a homogeneous linear system.

Alternatively, if $\dim(V) = n$, any set of $n$ linearly independent vectors automatically spans $V$, and any spanning set of $n$ vectors is automatically independent.

### The Standard Basis

The **standard basis** for $\mathbb{R}^n$ is:
$$\mathbf{e}_1 = (1, 0, 0, \ldots, 0)$$
$$\mathbf{e}_2 = (0, 1, 0, \ldots, 0)$$
$$\vdots$$
$$\mathbf{e}_n = (0, 0, 0, \ldots, 1)$$

Every vector $\mathbf{v} = (v_1, v_2, \ldots, v_n)$ is expressed as $\mathbf{v} = v_1\mathbf{e}_1 + v_2\mathbf{e}_2 + \cdots + v_n\mathbf{e}_n$.

The standard basis for $\mathcal{P}_n$ (polynomials of degree $\leq n$) is $\{1, x, x^2, \ldots, x^n\}$.

The standard basis for $\mathbb{R}^{m \times n}$ consists of the $mn$ matrices with a single 1 in one entry and 0 elsewhere.

### Coordinates Relative to a Basis

If $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ is a basis for $V$, then the coordinate vector $[\mathbf{v}]_{\mathcal{B}}$ is the unique $n$-tuple $(c_1, \ldots, c_n)^T$ such that $\mathbf{v} = c_1\mathbf{b}_1 + \cdots + c_n\mathbf{b}_n$.

To find coordinates, solve a linear system. For $\mathbb{R}^n$, if $\mathcal{B}$ is given as column vectors, solving $B\mathbf{c} = \mathbf{v}$ (where $B$ is the matrix with columns $\mathbf{b}_i$) yields $[\mathbf{v}]_{\mathcal{B}} = B^{-1}\mathbf{v}$.

### Change of Basis

Let $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ and $\mathcal{C} = \{\mathbf{c}_1, \ldots, \mathbf{c}_n\}$ be two bases for $V$. The **change-of-basis matrix** from $\mathcal{B}$ to $\mathcal{C}$, denoted $P_{\mathcal{C} \leftarrow \mathcal{B}}$, satisfies:
$$[\mathbf{v}]_{\mathcal{C}} = P_{\mathcal{C} \leftarrow \mathcal{B}} [\mathbf{v}]_{\mathcal{B}}$$

The $j$-th column of $P_{\mathcal{C} \leftarrow \mathcal{B}}$ is $[\mathbf{b}_j]_{\mathcal{C}}$, the coordinates of $\mathbf{b}_j$ in the $\mathcal{C}$ basis.

If $V = \mathbb{R}^n$ and we use the standard basis $\mathcal{E}$, then $P_{\mathcal{E} \leftarrow \mathcal{B}} = B$ (the matrix with columns $\mathbf{b}_i$). The matrix $B$ converts $\mathcal{B}$-coordinates to standard coordinates. Conversely, $B^{-1}$ converts standard coordinates to $\mathcal{B}$-coordinates.

### Dimension

The dimension of a vector space is the number of vectors in any basis. Key facts:

- If $\dim(V) = n$, any linearly independent set of $n$ vectors spans $V$.
- Any spanning set of $n$ vectors is linearly independent.
- Any linearly independent set can be extended to a basis.
- Any spanning set can be reduced to a basis.
- $\dim(\mathbb{R}^n) = n$
- $\dim(\mathbb{R}^{m \times n}) = mn$
- $\dim(\mathcal{P}_n) = n + 1$

## Formula(s)

**Coordinate representation**:
$$\mathbf{v} = \sum_{i=1}^{n} c_i \mathbf{b}_i = c_1 \mathbf{b}_1 + c_2 \mathbf{b}_2 + \cdots + c_n \mathbf{b}_n$$

**Change-of-basis matrix** (from $\mathcal{B}$ to $\mathcal{C}$):
$$P_{\mathcal{C} \leftarrow \mathcal{B}} = \begin{pmatrix} [\mathbf{b}_1]_{\mathcal{C}} & [\mathbf{b}_2]_{\mathcal{C}} & \cdots & [\mathbf{b}_n]_{\mathcal{C}} \end{pmatrix}$$

**Coordinate transformation**:
$$[\mathbf{v}]_{\mathcal{C}} = P_{\mathcal{C} \leftarrow \mathcal{B}} [\mathbf{v}]_{\mathcal{B}}$$

**Inverse transformation**:
$$[\mathbf{v}]_{\mathcal{B}} = P_{\mathcal{B} \leftarrow \mathcal{C}} [\mathbf{v}]_{\mathcal{C}} = (P_{\mathcal{C} \leftarrow \mathcal{B}})^{-1} [\mathbf{v}]_{\mathcal{C}}$$

## Properties

1. **Uniqueness of representation**: Every vector has exactly one coordinate representation in a given basis.
2. **Dimension invariance**: All bases of a finite-dimensional vector space have the same number of vectors.
3. **Basis extension theorem**: Any linearly independent set can be extended to a basis.
4. **Basis reduction theorem**: Any spanning set can be reduced to a basis.
5. **Dimension of a subspace**: If $W \subseteq V$, then $\dim(W) \leq \dim(V)$, with equality iff $W = V$.
6. **Rank-nullity theorem**: For a linear transformation $T: V \to W$, $\dim(\text{ker}(T)) + \dim(\text{im}(T)) = \dim(V)$.
7. **Change of basis is invertible**: $P_{\mathcal{C} \leftarrow \mathcal{B}}^{-1} = P_{\mathcal{B} \leftarrow \mathcal{C}}$.
8. **Matrix similarity**: Matrices $A$ and $B$ represent the same linear transformation in different bases iff $B = P^{-1}AP$ for some invertible $P$.
9. **Orthonormal basis**: If a basis is orthonormal ($\langle \mathbf{b}_i, \mathbf{b}_j \rangle = \delta_{ij}$), coordinates are given by inner products: $c_i = \langle \mathbf{v}, \mathbf{b}_i \rangle$.
10. **Dimension of sum**: $\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$.

## Step-by-Step Worked Examples

### Example 1: Verifying a Basis for $\mathbb{R}^2$

**Problem**: Determine whether $\mathcal{B} = \{(1, 2), (3, 4)\}$ is a basis for $\mathbb{R}^2$.

**Solution**:

Since $\dim(\mathbb{R}^2) = 2$, it suffices to check linear independence.

Vectors $\mathbf{v}_1 = (1, 2)$ and $\mathbf{v}_2 = (3, 4)$ are independent if $a\mathbf{v}_1 + b\mathbf{v}_2 = \mathbf{0}$ implies $a = b = 0$.

Set $a(1, 2) + b(3, 4) = (0, 0)$:
$$(a + 3b, 2a + 4b) = (0, 0)$$

This gives the system:
$$a + 3b = 0$$
$$2a + 4b = 0$$

From the first equation: $a = -3b$. Substitute into the second:
$$2(-3b) + 4b = -6b + 4b = -2b = 0 \implies b = 0$$
Then $a = -3(0) = 0$.

Thus $a = b = 0$, so the vectors are linearly independent. Since $\dim(\mathbb{R}^2) = 2$, $\mathcal{B}$ is a basis.

**Check spanning**: We can also verify that any vector $(x, y) \in \mathbb{R}^2$ can be expressed. Solve:
$$a(1, 2) + b(3, 4) = (x, y)$$
$$a + 3b = x$$
$$2a + 4b = y$$

From the first: $a = x - 3b$. Substitute:
$$2(x - 3b) + 4b = 2x - 6b + 4b = 2x - 2b = y$$
$$-2b = y - 2x \implies b = x - \frac{y}{2}$$
$$a = x - 3\left(x - \frac{y}{2}\right) = x - 3x + \frac{3y}{2} = -2x + \frac{3y}{2}$$

Thus any $(x, y) = \left(-2x + \frac{3y}{2}\right)(1, 2) + \left(x - \frac{y}{2}\right)(3, 4)$. Spanning holds.

### Example 2: Coordinates in $\mathcal{P}_2$

**Problem**: Find the coordinates of $p(x) = 2 + 3x - x^2$ relative to the basis $\mathcal{B} = \{1 + x, 1 - x, x + x^2\}$ for $\mathcal{P}_2$.

**Solution**:

We need scalars $c_1, c_2, c_3$ such that:
$$c_1(1 + x) + c_2(1 - x) + c_3(x + x^2) = 2 + 3x - x^2$$

Expand:
$$(c_1 + c_2) + (c_1 - c_2 + c_3)x + (c_3)x^2 = 2 + 3x - x^2$$

Equate coefficients:
Constant: $c_1 + c_2 = 2$
$x$: $c_1 - c_2 + c_3 = 3$
$x^2$: $c_3 = -1$

From $c_3 = -1$:
$c_1 - c_2 - 1 = 3 \implies c_1 - c_2 = 4$
$c_1 + c_2 = 2$

Adding: $2c_1 = 6 \implies c_1 = 3$
Then $c_2 = 2 - 3 = -1$

Thus $[p(x)]_{\mathcal{B}} = \begin{pmatrix} 3 \\ -1 \\ -1 \end{pmatrix}$.

Check: $3(1+x) + (-1)(1-x) + (-1)(x + x^2) = 3 + 3x - 1 + x - x - x^2 = 2 + 3x - x^2$. ✓

### Example 3: Change of Basis in $\mathbb{R}^2$

**Problem**: Let $\mathcal{B} = \{(1, 1), (1, -1)\}$ and $\mathcal{C} = \{(2, 0), (0, 1)\}$ be two bases for $\mathbb{R}^2$. Find the change-of-basis matrix $P_{\mathcal{C} \leftarrow \mathcal{B}}$.

**Solution**:

The $j$-th column of $P_{\mathcal{C} \leftarrow \mathcal{B}}$ is $[\mathbf{b}_j]_{\mathcal{C}}$.

**Column 1**: Express $\mathbf{b}_1 = (1, 1)$ in $\mathcal{C}$.
Solve $c_1(2, 0) + c_2(0, 1) = (1, 1)$:
$$(2c_1, c_2) = (1, 1) \implies c_1 = \frac{1}{2}, c_2 = 1$$
Thus $[\mathbf{b}_1]_{\mathcal{C}} = \begin{pmatrix} 1/2 \\ 1 \end{pmatrix}$.

**Column 2**: Express $\mathbf{b}_2 = (1, -1)$ in $\mathcal{C}$.
Solve $c_1(2, 0) + c_2(0, 1) = (1, -1)$:
$$(2c_1, c_2) = (1, -1) \implies c_1 = \frac{1}{2}, c_2 = -1$$
Thus $[\mathbf{b}_2]_{\mathcal{C}} = \begin{pmatrix} 1/2 \\ -1 \end{pmatrix}$.

Therefore:
$$P_{\mathcal{C} \leftarrow \mathcal{B}} = \begin{pmatrix} 1/2 & 1/2 \\ 1 & -1 \end{pmatrix}$$

**Verification**: Let $\mathbf{v} = (2, 0)$ which has $\mathcal{B}$-coordinates $[\mathbf{v}]_{\mathcal{B}} = (1, 1)^T$ (since $1(1,1) + 1(1,-1) = (2, 0)$). Then:
$$[\mathbf{v}]_{\mathcal{C}} = P_{\mathcal{C} \leftarrow \mathcal{B}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/2 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

Indeed, $(2, 0) = 1(2, 0) + 0(0, 1)$, so $[\mathbf{v}]_{\mathcal{C}} = (1, 0)^T$. ✓

### Example 4: Basis for a Subspace

**Problem**: Find a basis for the subspace $W = \{(x, y, z) \in \mathbb{R}^3 \mid x + 2y - z = 0\}$.

**Solution**:

The constraint $x + 2y - z = 0$ gives $z = x + 2y$. Any vector in $W$ can be written as:
$$(x, y, x + 2y) = x(1, 0, 1) + y(0, 1, 2)$$

Thus $W = \text{span}\{(1, 0, 1), (0, 1, 2)\}$.

Check linear independence: $a(1, 0, 1) + b(0, 1, 2) = (0, 0, 0)$ gives $a = 0$, $b = 0$, $a + 2b = 0$, which implies $a = b = 0$. So they are independent.

Therefore $\mathcal{B} = \{(1, 0, 1), (0, 1, 2)\}$ is a basis for $W$, and $\dim(W) = 2$.

### Example 5: Extending to a Basis

**Problem**: Extend the set $\{(1, 2, 1), (2, 1, 0)\}$ to a basis for $\mathbb{R}^3$.

**Solution**:

We need a third vector $\mathbf{v}_3$ such that the three vectors are linearly independent.

Method: Form a matrix with the given vectors as rows and augment with rows of the standard basis, then reduce to find which standard basis vectors are independent of the given set.

Place vectors as rows and row-reduce:
$$\begin{pmatrix} 1 & 2 & 1 \\ 2 & 1 & 0 \end{pmatrix}$$

Row reduce:
$$R_2 \leftarrow R_2 - 2R_1: \begin{pmatrix} 1 & 2 & 1 \\ 0 & -3 & -2 \end{pmatrix}$$

Row reduce to RREF:
$$R_2 \leftarrow -\frac{1}{3}R_2: \begin{pmatrix} 1 & 2 & 1 \\ 0 & 1 & 2/3 \end{pmatrix}$$
$$R_1 \leftarrow R_1 - 2R_2: \begin{pmatrix} 1 & 0 & -1/3 \\ 0 & 1 & 2/3 \end{pmatrix}$$

The pivot columns are 1 and 2. We need a vector whose third component is non-zero in the reduced form, but it's easier: take $\mathbf{e}_3 = (0, 0, 1)$.

Check independence of $\{(1, 2, 1), (2, 1, 0), (0, 0, 1)\}$:
$$\det\begin{pmatrix} 1 & 2 & 1 \\ 2 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} = 1 \cdot \det\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = 1 \cdot (1 - 4) = -3 \neq 0$$

Thus they are independent, and since $\dim(\mathbb{R}^3) = 3$, they form a basis.

### Example 6: Basis for Nullspace

**Problem**: Find a basis for the nullspace of $A = \begin{pmatrix} 1 & 2 & -1 \\ 2 & 4 & -2 \end{pmatrix}$.

**Solution**:

The nullspace is $\{\mathbf{x} \in \mathbb{R}^3 \mid A\mathbf{x} = \mathbf{0}\}$.

Solve $A\mathbf{x} = \mathbf{0}$:
$$x_1 + 2x_2 - x_3 = 0$$
$$2x_1 + 4x_2 - 2x_3 = 0$$

The second equation is twice the first, so it's redundant. We have:
$$x_1 = -2x_2 + x_3$$

Let $x_2 = s$, $x_3 = t$ (free variables). Then $x_1 = -2s + t$.
$$\mathbf{x} = \begin{pmatrix} -2s + t \\ s \\ t \end{pmatrix} = s\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix} + t\begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}$$

Thus $\left\{\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}\right\}$ is a basis for the nullspace. The nullspace has dimension $2$.

## Visual Interpretation

In $\mathbb{R}^2$, a basis is a pair of non-collinear vectors. Two vectors $\mathbf{v}_1$ and $\mathbf{v}_2$ form a basis if they are not parallel (i.e., one is not a scalar multiple of the other). Geometrically, they define a parallelogram grid covering the entire plane. The coordinates $(c_1, c_2)$ tell you how many steps of $\mathbf{v}_1$ and $\mathbf{v}_2$ to take to reach a point.

In $\mathbb{R}^3$, a basis is a set of three non-coplanar vectors. They define a parallelepiped grid filling the space. Visually, the standard basis corresponds to the usual $x$, $y$, $z$ axes. Other bases correspond to skewed coordinate systems.

A change of basis is a linear transformation that maps one coordinate grid onto another. This is equivalent to applying an invertible matrix that shears, rotates, or scales space without collapsing it. The determinant of the change-of-basis matrix gives the volume scaling factor between the basis parallelepipeds.

## Common Mistakes

1. **Confusing a basis with a spanning set**: A spanning set must be linearly independent to be a basis. The set $\{(1,0), (0,1), (1,1)\}$ spans $\mathbb{R}^2$ but is not a basis because it is not independent.

2. **Assuming the standard basis is the only basis**: There are infinitely many bases for any non-trivial vector space. Different bases are useful for different purposes.

3. **Forgetting to check both spanning and independence**: Either condition alone is insufficient. A linearly independent set might not span; a spanning set might not be independent.

4. **Miscomputing coordinates**: Coordinates must be taken relative to a specific, ordered basis. Changing the order changes the coordinate vector.

5. **Incorrect dimension counting**: $\dim(\mathbb{R}^{m \times n}) = mn$, not $m + n$. The space of $3 \times 3$ matrices has dimension 9.

6. **Confusing the change-of-basis matrix with its inverse**: The matrix $P_{\mathcal{C} \leftarrow \mathcal{B}}$ converts $\mathcal{B}$-coordinates to $\mathcal{C}$-coordinates, not the other way. Getting the direction wrong is a very common error.

7. **Assuming every linearly independent set can be extended to a basis in the same way**: The extension must be done systematically (e.g., by checking which standard basis vectors are independent of the given set).

8. **Thinking bases must be orthogonal**: A basis does not require orthogonality. Any set of $\dim(V)$ linearly independent vectors forms a basis, regardless of angles between them.

9. **Treating $\mathbb{R}^2$ as a subset of $\mathbb{R}^3$**: When finding a basis for a subspace of $\mathbb{R}^3$, the basis vectors must be in $\mathbb{R}^3$, not $\mathbb{R}^2$.

10. **Ignoring the field**: The field matters for dimension. $\mathbb{R}$ is 1-dimensional as a vector space over $\mathbb{R}$ but infinite-dimensional over $\mathbb{Q}$.

## Interview Questions

### Beginner

**Q1**: What is a basis of a vector space?

**A**: A basis is a set of vectors that is linearly independent and spans the vector space. Every vector can be written uniquely as a linear combination of basis vectors.

**Q2**: What is the standard basis of $\mathbb{R}^3$?

**A**: The standard basis is $\{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}$. These are the unit vectors along the $x$, $y$, and $z$ axes.

**Q3**: How do you find the coordinates of a vector relative to a basis?

**A**: Solve a linear system. For $\mathbb{R}^n$ with basis $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$, solve $c_1\mathbf{b}_1 + \cdots + c_n\mathbf{b}_n = \mathbf{v}$ for $(c_1, \ldots, c_n)$. This is $B\mathbf{c} = \mathbf{v}$ where $B$ has columns $\mathbf{b}_i$.

**Q4**: What is the dimension of $\mathbb{R}^4$?

**A**: 4.

**Q5**: Is $\{(1, 0), (2, 0)\}$ a basis for $\mathbb{R}^2$?

**A**: No. They are linearly dependent: $(2, 0) = 2(1, 0)$. A basis for $\mathbb{R}^2$ requires two linearly independent vectors.

### Intermediate

**Q1**: Find a basis for the subspace $W = \{(x, y, z) \mid x - y + 2z = 0\}$ of $\mathbb{R}^3$.

**A**: From $x = y - 2z$, any vector is $(y - 2z, y, z) = y(1, 1, 0) + z(-2, 0, 1)$. Basis: $\{(1, 1, 0), (-2, 0, 1)\}$, dimension 2.

**Q2**: Explain the change-of-basis matrix and how to compute it.

**A**: $P_{\mathcal{C} \leftarrow \mathcal{B}}$ converts $\mathcal{B}$-coordinates to $\mathcal{C}$-coordinates. Its $j$-th column is $[\mathbf{b}_j]_{\mathcal{C}}$, the coordinates of the $j$-th $\mathcal{B}$ basis vector expressed in $\mathcal{C}$. If $\mathcal{E}$ is the standard basis, $P_{\mathcal{E} \leftarrow \mathcal{B}} = B$ (matrix of $\mathcal{B}$ vectors as columns), and $P_{\mathcal{B} \leftarrow \mathcal{E}} = B^{-1}$.

**Q3**: Prove that if a vector space has a basis of $n$ vectors, every basis has $n$ vectors.

**A**: The Steinitz exchange lemma: if $\mathcal{B}_1$ has $m$ vectors and $\mathcal{B}_2$ has $n$ vectors, then $m \leq n$ and $n \leq m$, so $m = n$. Intuitively, spanning sets cannot be smaller than independent sets, and a basis is both, so all bases have the same size.

**Q4**: What is the dimension of the space of $n \times n$ symmetric matrices?

**A**: $\frac{n(n+1)}{2}$. A symmetric matrix has $n$ diagonal entries and $\frac{n(n-1)}{2}$ independent off-diagonal entries, totaling $n + \frac{n(n-1)}{2} = \frac{n(n+1)}{2}$.

**Q5**: How does PCA relate to the concept of a basis?

**A**: PCA finds an orthonormal basis (the eigenvectors of the covariance matrix) ordered by eigenvalue magnitude. The first basis vector (first PC) captures the most variance. This basis is ideal for dimensionality reduction: projecting onto the first $k$ PCs gives the best $k$-dimensional linear approximation of the data.

### Advanced

**Q1**: Every vector space has a basis. Discuss this statement for infinite-dimensional spaces.

**A**: For finite-dimensional spaces, a basis always exists (by successive extension of a linearly independent set). For infinite-dimensional spaces, the existence of a basis requires Zorn's Lemma (equivalent to the Axiom of Choice). The basis is then called a Hamel basis. For example, $\mathbb{R}$ as a vector space over $\mathbb{Q}$ has a Hamel basis, but it is uncountable and cannot be explicitly constructed. In functional analysis, the more useful notion is a Schauder basis (a countable basis where vectors are represented as infinite series rather than finite linear combinations).

**Q2**: Let $V$ be an $n$-dimensional vector space and $T: V \to V$ a linear transformation. Prove that if $\mathbf{v}, T(\mathbf{v}), \ldots, T^{n-1}(\mathbf{v})$ are linearly independent, then they form a basis for $V$ and the matrix of $T$ in this basis is a companion matrix.

**A**: Since there are $n$ independent vectors and $\dim(V) = n$, they form a basis $\mathcal{B} = \{\mathbf{v}, T(\mathbf{v}), \ldots, T^{n-1}(\mathbf{v})\}$. Write $T^n(\mathbf{v})$ as a linear combination: $T^n(\mathbf{v}) = -a_0\mathbf{v} - a_1T(\mathbf{v}) - \cdots - a_{n-1}T^{n-1}(\mathbf{v})$. Then the matrix of $T$ in $\mathcal{B}$ is:
$$\begin{pmatrix} 0 & 0 & \cdots & 0 & -a_0 \\ 1 & 0 & \cdots & 0 & -a_1 \\ 0 & 1 & \cdots & 0 & -a_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & 1 & -a_{n-1} \end{pmatrix}$$
This is the companion matrix of the polynomial $p(\lambda) = \lambda^n + a_{n-1}\lambda^{n-1} + \cdots + a_0$.

**Q3**: Explain the role of basis functions in kernel methods. How does the representer theorem relate to the basis of an RKHS?

**A**: In an RKHS $\mathcal{H}$ with kernel $K$, the functions $\{K(\cdot, \mathbf{x}_i)\}$ form a basis for the subspace of $\mathcal{H}$ spanned by the training data. The representer theorem states that the optimal solution to a regularized empirical risk minimization problem in $\mathcal{H}$ lies in the span of $\{K(\cdot, \mathbf{x}_i)\}_{i=1}^n$, so $f^*(\mathbf{x}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}, \mathbf{x}_i)$. This reduces an infinite-dimensional optimization to a finite-dimensional one. The basis functions $K(\cdot, \mathbf{x}_i)$ are called kernel basis functions.

## Practice Problems

### Easy — 5 Questions

**E1**: Is $\{(1, 1), (2, 3)\}$ a basis for $\mathbb{R}^2$?

**E2**: Find the coordinates of $(3, -2)$ relative to the standard basis of $\mathbb{R}^2$.

**E3**: What is the dimension of $\mathcal{P}_3$ (polynomials of degree at most 3)?

**E4**: Find a basis for $\mathbb{R}^2$ that includes $(1, 2)$.

**E5**: True or False: $\{(1, 0, 0), (0, 1, 0), (1, 1, 0)\}$ is a basis for $\mathbb{R}^3$.

### Medium — 5 Questions

**M1**: Find a basis for the subspace of $\mathbb{R}^3$ spanned by $\{(1, 1, 0), (2, 2, 0), (0, 0, 1)\}$. What is its dimension?

**M2**: Let $\mathcal{B} = \{(1, 1), (0, 2)\}$ be a basis for $\mathbb{R}^2$. Find $[(5, -2)]_{\mathcal{B}}$.

**M3**: Find the change-of-basis matrix from $\mathcal{B} = \{(1, 0), (1, 1)\}$ to $\mathcal{C} = \{(2, 1), (1, 2)\}$.

**M4**: Determine the dimension of the space of $2 \times 2$ matrices with zero trace, and find a basis.

**M5**: Extend $\{1 + x, 1 - x\}$ to a basis for $\mathcal{P}_2$.

### Hard — 3 Questions

**H1**: Let $V$ be the set of all sequences $(x_n)$ satisfying $x_{n+2} = x_{n+1} + x_n$ (Fibonacci-like sequences). Find a basis and prove it is a basis.

**H2**: Prove that if $U$ and $W$ are subspaces of $V$, then $\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$.

**H3**: Let $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ be a basis for $V$ and let $\mathbf{u}_i = \sum_{j=1}^n a_{ij} \mathbf{v}_j$. Prove that $\{\mathbf{u}_1, \ldots, \mathbf{u}_n\}$ is a basis iff the matrix $A = (a_{ij})$ is invertible.

## Solutions

### Easy Solutions

**E1**: Check independence: $a(1, 1) + b(2, 3) = (0, 0)$ gives $a + 2b = 0$, $a + 3b = 0$. Subtracting: $-b = 0 \implies b = 0$, then $a = 0$. They are independent. Since $\dim(\mathbb{R}^2) = 2$, they form a basis.

**E2**: $(3, -2) = 3(1, 0) + (-2)(0, 1)$, so coordinates are $(3, -2)^T$.

**E3**: $\mathcal{P}_3$ has basis $\{1, x, x^2, x^3\}$, dimension 4.

**E4**: Need a vector $\mathbf{v}$ such that $(1, 2)$ and $\mathbf{v}$ are independent. Choose $\mathbf{v} = (0, 1)$. Check: $\det\begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix} = 1 \neq 0$. So $\{(1, 2), (0, 1)\}$ is a basis.

**E5**: False. The three vectors lie in the $xy$-plane (third component is 0). They are linearly dependent: $(1, 1, 0) = (1, 0, 0) + (0, 1, 0)$. They do not span $\mathbb{R}^3$ (cannot reach points with non-zero $z$).

### Medium Solutions

**M1**: Vectors: $(1, 1, 0)$, $(2, 2, 0)$, $(0, 0, 1)$. Since $(2, 2, 0) = 2(1, 1, 0)$, it is redundant. The remaining vectors $(1, 1, 0)$ and $(0, 0, 1)$ are independent (they are not scalar multiples). They span the set $\{(x, x, z) \mid x, z \in \mathbb{R}\}$. Basis: $\{(1, 1, 0), (0, 0, 1)\}$, dimension 2.

**M2**: Solve $c_1(1, 1) + c_2(0, 2) = (5, -2)$:
$$c_1 = 5 \quad \text{(first component)}$$
$$c_1 + 2c_2 = -2 \quad \text{(second component)}$$
$$5 + 2c_2 = -2 \implies 2c_2 = -7 \implies c_2 = -\frac{7}{2}$$
Thus $[(5, -2)]_{\mathcal{B}} = \begin{pmatrix} 5 \\ -7/2 \end{pmatrix}$.

**M3**: Compute $P_{\mathcal{C} \leftarrow \mathcal{B}}$. Columns are $\mathcal{B}$ vectors in $\mathcal{C}$ coordinates.

Column 1: $(1, 0)$ in $\mathcal{C}$:
Solve $a(2, 1) + b(1, 2) = (1, 0)$:
$$2a + b = 1$$
$$a + 2b = 0 \implies a = -2b$$
Substitute: $2(-2b) + b = -4b + b = -3b = 1 \implies b = -1/3$, $a = 2/3$.
Thus $[(1, 0)]_{\mathcal{C}} = \begin{pmatrix} 2/3 \\ -1/3 \end{pmatrix}$.

Column 2: $(1, 1)$ in $\mathcal{C}$:
Solve $a(2, 1) + b(1, 2) = (1, 1)$:
$$2a + b = 1$$
$$a + 2b = 1 \implies a = 1 - 2b$$
Substitute: $2(1 - 2b) + b = 2 - 4b + b = 2 - 3b = 1 \implies b = 1/3$, $a = 1 - 2/3 = 1/3$.
Thus $[(1, 1)]_{\mathcal{C}} = \begin{pmatrix} 1/3 \\ 1/3 \end{pmatrix}$.

Therefore $P_{\mathcal{C} \leftarrow \mathcal{B}} = \begin{pmatrix} 2/3 & 1/3 \\ -1/3 & 1/3 \end{pmatrix}$.

**M4**: Skew-symmetric $2 \times 2$ matrices have form $\begin{pmatrix} 0 & a \\ -a & 0 \end{pmatrix}$. One basis vector: $\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$. Dimension = 1.

**M5**: Need a third polynomial $p(x)$ such that $\{1+x, 1-x, p(x)\}$ is a basis for $\mathcal{P}_2$ (dimension 3). Choose $p(x) = x^2$. Check independence: $a(1+x) + b(1-x) + c x^2 = 0 + 0x + 0x^2$ gives constant: $a + b = 0$, $x$: $a - b = 0$, $x^2$: $c = 0$. From $a+b = 0$ and $a-b = 0$, we get $a=b=0$. So they are independent, and since there are 3 in a 3-dimensional space, they form a basis.

### Hard Solutions

**H1**: $V = \{(x_1, x_2, x_3, \ldots) \mid x_{n+2} = x_{n+1} + x_n\}$.

Every sequence is determined by its first two terms. Write:
$$\mathbf{x} = (x_1, x_2, x_1 + x_2, x_1 + 2x_2, 2x_1 + 3x_2, \ldots)$$
$$= x_1(1, 0, 1, 1, 2, \ldots) + x_2(0, 1, 1, 2, 3, \ldots)$$

Let $\mathbf{f}_1 = (1, 0, 1, 1, 2, 3, 5, \ldots)$ (the Fibonacci numbers starting at $F_1=1, F_2=0$?) — actually, let's be precise: $\mathbf{f}_1$ is the sequence with $x_1 = 1, x_2 = 0$ satisfying $x_{n+2} = x_{n+1} + x_n$: $(1, 0, 1, 1, 2, 3, 5, \ldots)$.
$\mathbf{f}_2$ is the sequence with $x_1 = 0, x_2 = 1$: $(0, 1, 1, 2, 3, 5, 8, \ldots)$.

Independence: $a\mathbf{f}_1 + b\mathbf{f}_2 = \mathbf{0}$ means first two components are $a \cdot 1 + b \cdot 0 = 0$ and $a \cdot 0 + b \cdot 1 = 0$, so $a = b = 0$. Spanning: any sequence is $x_1\mathbf{f}_1 + x_2\mathbf{f}_2$. Thus $\{\mathbf{f}_1, \mathbf{f}_2\}$ is a basis, $\dim(V) = 2$.

**H2**: Let $\dim(U) = p$, $\dim(W) = q$, $\dim(U \cap W) = r$. Take a basis $\mathcal{B}_0 = \{\mathbf{v}_1, \ldots, \mathbf{v}_r\}$ for $U \cap W$. Extend to a basis $\mathcal{B}_U = \mathcal{B}_0 \cup \{\mathbf{u}_1, \ldots, \mathbf{u}_{p-r}\}$ for $U$ and $\mathcal{B}_W = \mathcal{B}_0 \cup \{\mathbf{w}_1, \ldots, \mathbf{w}_{q-r}\}$ for $W$.

Claim: $\mathcal{B} = \mathcal{B}_0 \cup \{\mathbf{u}_1, \ldots, \mathbf{u}_{p-r}\} \cup \{\mathbf{w}_1, \ldots, \mathbf{w}_{q-r}\}$ is a basis for $U + W$.

Spanning: Any $\mathbf{x} \in U + W$ is $\mathbf{x} = \mathbf{u} + \mathbf{w}$. Write $\mathbf{u}$ and $\mathbf{w}$ in their respective bases, combine.

Independence: Suppose sum is zero: $\sum a_i\mathbf{v}_i + \sum b_j\mathbf{u}_j + \sum c_k\mathbf{w}_k = \mathbf{0}$. Then $\sum c_k\mathbf{w}_k = -\sum a_i\mathbf{v}_i - \sum b_j\mathbf{u}_j \in U$. But $\sum c_k\mathbf{w}_k \in W$, so it's in $U \cap W$, hence can be written in $\mathcal{B}_0$. Since $\mathcal{B}_W$ is independent and $\mathcal{B}_0$ is a subset, this forces all $c_k = 0$. Similarly all $b_j = 0$, then all $a_i = 0$. So basis.

Number of elements: $r + (p - r) + (q - r) = p + q - r = \dim(U) + \dim(W) - \dim(U \cap W)$.

**H3**: $\{\mathbf{u}_1, \ldots, \mathbf{u}_n\}$ is a basis iff it is linearly independent. The coordinate vectors of $\mathbf{u}_i$ in basis $\mathcal{B}$ are the columns of $A$. Linear independence of $\{\mathbf{u}_i\}$ is equivalent to $\sum c_i \mathbf{u}_i = \mathbf{0} \implies$ all $c_i = 0$. But $\sum c_i \mathbf{u}_i = \sum c_i \sum_j a_{ij} \mathbf{v}_j = \sum_j (\sum_i c_i a_{ij}) \mathbf{v}_j = \mathbf{0}$.

Since $\mathcal{B}$ is a basis, each coefficient $\sum_i c_i a_{ij} = 0$. This is $A^T \mathbf{c} = \mathbf{0}$ (where $\mathbf{c} = (c_1, \ldots, c_n)^T$). The vectors are independent iff $A^T \mathbf{c} = \mathbf{0}$ has only the trivial solution, which occurs iff $A^T$ is invertible (i.e., $\det(A^T) \neq 0$), which is equivalent to $A$ being invertible.

## Related Concepts

- **Vector space (MATH-031)**: The abstract structure in which bases live.
- **Span (MATH-033)**: The set of all linear combinations; a basis is a spanning set that is also independent.
- **Linear independence**: A set is independent if no vector is a linear combination of the others.
- **Dimension**: The number of vectors in any basis; an invariant of the vector space.
- **Change of basis**: The linear transformation that converts coordinates from one basis to another.
- **Coordinate vector**: The representation of a vector in a specific basis.
- **Rank of a matrix**: The dimension of the column space, which equals the number of pivot columns.
- **Eigenbasis**: A basis consisting of eigenvectors of a linear transformation.

## Next Concepts

- Linear transformations and their matrix representations
- Eigenvalues and eigenvectors
- Diagonalization
- Inner product spaces and orthonormal bases (Gram-Schmidt)
- Singular value decomposition
- Principal component analysis

## Summary

A basis is a set of vectors that is both linearly independent and spans the entire vector space. Every vector has a unique coordinate representation in a given basis. The dimension of a vector space is the number of basis vectors and is invariant — all bases of a space have the same cardinality. Change of basis allows us to move between different coordinate systems for the same space, with the transition governed by an invertible matrix. Bases are the foundation for representing vectors and linear transformations concretely, making abstract vector spaces amenable to computation.

## Key Takeaways

1. A basis is a linearly independent spanning set — it is both sufficient (spans) and efficient (independent).
2. Every vector has unique coordinates in a given basis.
3. Dimension = number of vectors in any basis; it is well-defined and invariant.
4. The standard basis is the simplest basis, but many other bases exist and are useful.
5. Change-of-basis matrices convert coordinates between different bases; they are always invertible.
6. To check if a set is a basis: verify independence (or spanning) and check cardinality matches dimension.
7. PCA finds an optimal basis (eigenbasis) ordered by variance for dimensionality reduction.
8. Basis functions (polynomials, Fourier, RBFs) are fundamental in regression and signal processing.
9. The representer theorem shows that RKHS solutions live in the span of kernel basis functions.
10. Bases transform abstract vector spaces into concrete coordinate representations, enabling all of computational linear algebra.
