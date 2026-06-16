# Concept: Eigenvector

## Concept ID

MATH-040

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Define eigenvectors as non-zero vectors satisfying $A\mathbf{v} = \lambda \mathbf{v}$.
- Compute eigenvectors by solving $(A - \lambda I)\mathbf{v} = \mathbf{0}$ for each eigenvalue.
- Understand that eigenvectors are direction-invariant under the linear transformation.
- Relate eigenvectors to the eigenspace: $\ker(A - \lambda I)$.
- Explain orthogonality of eigenvectors for symmetric matrices.
- Construct an eigenbasis for a diagonalisable matrix.
- Apply eigenvectors in PCA, spectral clustering, PageRank, and LDA.

## Prerequisites

- Eigenvalues (MATH-039) — what they are and how to compute them.
- Solving homogeneous linear systems $(A - \lambda I)\mathbf{v} = \mathbf{0}$.
- Row reduction and parametric solution forms.
- Basic matrix algebra and vector spaces.
- Dot products and orthogonal complements.

## Definition

Let $A \in \mathbb{C}^{n \times n}$ be a square matrix. A non-zero vector $\mathbf{v} \in \mathbb{C}^{n}$ is called an **eigenvector** of $A$ associated with the eigenvalue $\lambda$ if:

\[
A \mathbf{v} = \lambda \mathbf{v}.
\]

The vector $\mathbf{v}$ is said to be an eigenvector **corresponding to** (or **belonging to**) the eigenvalue $\lambda$. The pair $(\lambda, \mathbf{v})$ is called an **eigenpair**.

Rearranging gives the homogeneous system:

\[
(A - \lambda I) \mathbf{v} = \mathbf{0}.
\]

Thus the eigenvectors for eigenvalue $\lambda$ are precisely the non-zero vectors in the nullspace of $A - \lambda I$, called the **eigenspace** corresponding to $\lambda$:

\[
E_\lambda = \ker(A - \lambda I) = \{\mathbf{v} \in \mathbb{C}^n : (A - \lambda I)\mathbf{v} = \mathbf{0}\}.
\]

The dimension of $E_\lambda$ is the **geometric multiplicity** of $\lambda$.

## Intuition

Imagine a linear transformation as a machine that stretches, rotates, and shears space. Most vectors change direction when fed through the machine. Eigenvectors are the special directions that remain on the same line — they are only scaled (or, if $\lambda$ is negative, reflected through the origin). The eigenvalue tells you the factor by which the eigenvector is stretched.

For a $2 \times 2$ matrix with two real eigenvalues, the eigenvectors give the axes of the ellipse that results from mapping the unit circle. These axes are the directions of maximal and minimal stretching. For a symmetric matrix, these axes are perpendicular.

If $\lambda = 0$, the eigenvector satisfies $A\mathbf{v} = \mathbf{0}$, meaning $\mathbf{v}$ is in the nullspace of $A$. The zero-eigenvalue eigenvectors are exactly the kernel.

## Why This Concept Matters

Eigenvectors reveal the "natural coordinate system" of a linear transformation. When you express a transformation in its eigenbasis (the basis of eigenvectors), the matrix becomes diagonal — the transformation reduces to independent scalings along each axis. This is the essence of **diagonalisation**, which simplifies:

- **Powers of a matrix**: $A^k = P D^k P^{-1}$ is trivial when eigenvectors are known.
- **Dynamical systems**: $\mathbf{x}_{k+1} = A \mathbf{x}_k$ decouples into independent scalar recurrences in the eigenbasis.
- **Dimensionality reduction**: The eigenvectors of the covariance matrix (principal components) point in the directions of maximum variance.
- **Graph analysis**: The eigenvectors of the graph Laplacian reveal clusters and communities.
- **Quantum mechanics**: The eigenvectors of an operator correspond to observable states with definite measurement values.

## Historical Background

The study of eigenvectors began with **Euler** and **Lagrange** in the 18th century, who studied principal axes of rigid body rotation. The inertia matrix of a rotating body has eigenvectors (principal axes) along which the angular momentum and angular velocity are aligned.

**Augustin-Louis Cauchy** (1789–1857) formalised the notion of eigenvectors in the context of the characteristic equation. He proved that symmetric matrices have real eigenvalues and orthogonal eigenvectors.

**James Joseph Sylvester** (1814–1897) developed the law of inertia for quadratic forms, which states that the number of positive, negative, and zero eigenvalues (and their eigenvectors) is invariant under congruence transformations.

**David Hilbert** and **Erhard Schmidt** extended eigenvector theory to infinite-dimensional spaces (Hilbert spaces), founding spectral theory. **John von Neumann** applied this framework to quantum mechanics, where eigenvectors represent states and eigenvalues represent measurement outcomes.

## Real World Examples

### 1. Google PageRank

The PageRank algorithm models web surfing as a random walk on the directed graph of web pages. The PageRank vector is the stationary distribution of this Markov chain, which is the eigenvector of the Google matrix $G$ corresponding to eigenvalue $\lambda = 1$. Power iteration multiplies $G$ repeatedly to find this dominant eigenvector. Google's original paper reported convergence in about 50 iterations on the 2004 web graph.

### 2. Facial Recognition — Eigenfaces

In face recognition, each face image is vectorised. PCA on the face dataset yields eigenvectors (called "eigenfaces") of the covariance matrix. A new face is projected onto the top $k$ eigenfaces, and recognition is performed in this $k$-dimensional subspace. The eigenfaces capture the modes of variation in the dataset (lighting, expression, identity direction).

### 3. Vibration Analysis — Mode Shapes

In mechanical engineering, each natural frequency (eigenvalue) of a structure has an associated mode shape (eigenvector). When a building vibrates during an earthquake, the displacement pattern follows a linear combination of these mode shapes. The first mode (lowest eigenvalue) typically dominates.

### 4. Principal Component Analysis

In PCA, the eigenvectors of the covariance matrix $S = \frac{1}{N} X^{\mathsf{T}} X$ are called **principal directions**. The first principal component $\mathbf{v}_1$ (eigenvector with largest eigenvalue) points in the direction of maximum variance in the data. Projecting data onto $\mathbf{v}_1$ gives the 1D representation that preserves the most information.

### 5. Spectral Clustering

Given a similarity graph of data points, the eigenvectors of the graph Laplacian $L = D - A$ are used to embed the data into $\mathbb{R}^{k}$ (using the $k$ eigenvectors corresponding to the smallest non-zero eigenvalues). This embedding often reveals cluster structure that is not linearly separable in the original space.

## AI/ML Relevance

### 1. PCA and Dimensionality Reduction

Given centred data $X \in \mathbb{R}^{N \times d}$, the sample covariance matrix is $S = \frac{1}{N} X^{\mathsf{T}} X$. The top $k$ eigenvectors of $S$ (those with largest eigenvalues) form the projection matrix $W \in \mathbb{R}^{d \times k}$. The reduced representation is $Z = X W \in \mathbb{R}^{N \times k}$. Each column of $W$ is an eigenvector pointing in a direction of maximal residual variance.

Concrete example: On the MNIST dataset ($28 \times 28 = 784$ dimensions), the top 10 eigenvectors of the covariance matrix capture smooth handwritten stroke patterns. The eigenvalue decay shows that about 100 components explain 95% of the variance.

### 2. Spectral Clustering

Construct the graph Laplacian $L = D - A$. Compute the $k$ eigenvectors corresponding to the $k$ smallest non-zero eigenvalues. Form $U \in \mathbb{R}^{N \times k}$ with these eigenvectors as columns. Cluster the rows of $U$ using $k$-means. This works because the eigenvectors of $L$ encode the connected components of the graph — for $k$ disconnected components, the first $k$ eigenvectors are indicator vectors of the clusters.

### 3. PageRank and Network Centrality

The eigenvector centrality of a node in a graph is the corresponding entry of the dominant eigenvector of the adjacency matrix. Nodes with high eigenvector centrality are connected to other influential nodes. PageRank modifies this by adding a damping factor to handle disconnected regions and dangling nodes.

### 4. Fisher's Linear Discriminant (LDA)

Linear Discriminant Analysis finds eigenvectors of $S_W^{-1} S_B$, where $S_W$ is the within-class scatter matrix and $S_B$ is the between-class scatter matrix. The eigenvectors (called Fisherfaces in face recognition) define the projection directions that maximise class separability. For a $K$-class problem, there are $K-1$ meaningful discriminant directions.

### 5. Spectral Normalisation in GANs

In spectral normalisation, the weight matrix $W$ of each layer is divided by its spectral norm $\|W\|_2 = \sigma_{\max}(W)$, the largest singular value. The right singular vector (eigenvector of $W^{\mathsf{T}} W$) corresponding to $\sigma_{\max}$ indicates the direction that is most amplified by the layer. Constraining this to $1$ prevents the discriminator from becoming too steep.

### 6. Eigenvector-Following in Optimisation

In some optimisation methods (e.g., the Levenberg–Marquardt algorithm), the eigenvectors of the Hessian indicate directions of different curvatures. The eigendirections with small eigenvalues correspond to flat directions where the objective changes slowly; the algorithm takes larger steps in these directions.

### 7. Collaborative Filtering and Matrix Factorisation

In matrix factorisation for recommender systems, the latent factors (columns of $U$ and $V$ in $R \approx UV^{\mathsf{T}}$) can be interpreted as eigenvectors of the user-user and item-item similarity matrices. The top eigenvectors capture the dominant patterns in user preferences.

## Mathematical Explanation

### Computing Eigenvectors

Once an eigenvalue $\lambda$ is known, the eigenvectors are found by solving $(A - \lambda I)\mathbf{v} = \mathbf{0}$. This is a homogeneous linear system:

\[
(A - \lambda I)\mathbf{v} = 0.
\]

Row-reduce $A - \lambda I$ to RREF, identify free variables, and express the solution parametrically.

For an $n \times n$ matrix, if $\lambda$ has geometric multiplicity $g$, the solution space has dimension $g$, and we can find $g$ linearly independent eigenvectors spanning $E_\lambda$.

### Example: $2 \times 2$ Case

Let $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$ with eigenvalues $\lambda_1 = 2$, $\lambda_2 = 5$ (from MATH-039).

For $\lambda = 2$:
\[
A - 2I = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & \tfrac12 \\ 0 & 0 \end{bmatrix}.
\]
So $v_1 + \frac12 v_2 = 0$, giving $\mathbf{v} = t\begin{bmatrix} 1 \\ -2 \end{bmatrix}$ for $t \neq 0$.

For $\lambda = 5$:
\[
A - 5I = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]
So $v_1 - v_2 = 0$, giving $\mathbf{v} = t\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ for $t \neq 0$.

The matrix has two linearly independent eigenvectors, so it is diagonalisable.

### Orthogonality of Eigenvectors for Symmetric Matrices

If $A$ is a real symmetric matrix ($A^{\mathsf{T}} = A$), then eigenvectors corresponding to distinct eigenvalues are orthogonal. Proof:

Let $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$.

\[
\lambda_1 \mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2 = (A\mathbf{v}_1)^{\mathsf{T}} \mathbf{v}_2 = \mathbf{v}_1^{\mathsf{T}} A^{\mathsf{T}} \mathbf{v}_2 = \mathbf{v}_1^{\mathsf{T}} A \mathbf{v}_2 = \lambda_2 \mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2.
\]

Thus $(\lambda_1 - \lambda_2)(\mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2) = 0$. Since $\lambda_1 \neq \lambda_2$, we have $\mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2 = 0$, i.e., the eigenvectors are orthogonal.

### Eigenspaces and Geometric Multiplicity

The eigenspace $E_\lambda$ is a subspace of $\mathbb{C}^n$ (or $\mathbb{R}^n$ if $\lambda$ is real). Its dimension equals the number of free variables in $(A - \lambda I)\mathbf{v} = \mathbf{0}$, which is $n - \operatorname{rank}(A - \lambda I)$.

If geometric multiplicity equals algebraic multiplicity for every eigenvalue, the matrix is **diagonalisable** — there exists a basis of $\mathbb{R}^n$ consisting entirely of eigenvectors.

## Formula(s)

\[
A \mathbf{v} = \lambda \mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}
\]

\[
(A - \lambda I) \mathbf{v} = \mathbf{0}
\]

\[
E_\lambda = \ker(A - \lambda I) = \{\mathbf{v} \in \mathbb{C}^n : (A - \lambda I)\mathbf{v} = \mathbf{0}\}
\]

\[
\dim(E_\lambda) = \text{geometric multiplicity of } \lambda
\]

\[
\mathbf{v}_i^{\mathsf{T}} \mathbf{v}_j = 0 \quad \text{for } \lambda_i \neq \lambda_j \text{ and } A^{\mathsf{T}} = A
\]

## Properties

1. **Non-uniqueness**: If $\mathbf{v}$ is an eigenvector, so is $c\mathbf{v}$ for any non-zero scalar $c$. Eigenvectors are only defined up to scaling. It is common to normalise them to unit length.

2. **Linear independence**: Eigenvectors corresponding to distinct eigenvalues are linearly independent.

3. **Eigenspace**: The set of all eigenvectors for $\lambda$, together with the zero vector, forms a subspace $E_\lambda$.

4. **Spectral theorem**: For a real symmetric $A$, there exists an orthonormal basis of $\mathbb{R}^n$ consisting of eigenvectors of $A$.

5. **Defective matrices**: If the geometric multiplicity is less than the algebraic multiplicity, the matrix is defective — it lacks a full set of eigenvectors and cannot be diagonalised.

6. **Relation to the nullspace**: Eigenvectors for $\lambda = 0$ are exactly the non-zero vectors in $\ker(A)$.

7. **Eigenvectors of $A^{-1}$**: If $A$ is invertible, then $A^{-1}$ has the same eigenvectors as $A$, with eigenvalues $1/\lambda$.

8. **Eigenvectors of $A^k$**: $\mathbf{v}$ is an eigenvector of $A^k$ (same $\mathbf{v}$, eigenvalue $\lambda^k$).

9. **Eigenvectors of $p(A)$**: $\mathbf{v}$ is an eigenvector of $p(A)$ for any polynomial $p$, with eigenvalue $p(\lambda)$.

10. **Left eigenvectors**: A row vector $\mathbf{w}^{\mathsf{T}}$ satisfying $\mathbf{w}^{\mathsf{T}} A = \lambda \mathbf{w}^{\mathsf{T}}$ is a left eigenvector. Left eigenvectors are right eigenvectors of $A^{\mathsf{T}}$.

## Step-by-Step Worked Examples

### Example 1: Finding Eigenvectors of a $2 \times 2$ Matrix

Find the eigenvectors of $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$.

**Step 1**: Find eigenvalues.

\[
\det\begin{bmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{bmatrix} = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 2)(\lambda - 4) = 0.
\]

So $\lambda_1 = 2$, $\lambda_2 = 4$.

**Step 2**: Eigenspace for $\lambda_1 = 2$.

\[
A - 2I = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}.
\]

Equation: $v_1 + v_2 = 0$, so $v_2 = -v_1$. Parametric: $\mathbf{v} = t \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

One eigenvector: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

**Step 3**: Eigenspace for $\lambda_2 = 4$.

\[
A - 4I = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]

Equation: $v_1 - v_2 = 0$, so $v_1 = v_2$. Parametric: $\mathbf{v} = t \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

One eigenvector: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

**Step 4**: Note that $\mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2 = 1 \cdot 1 + (-1) \cdot 1 = 0$. Since $A$ is symmetric, the eigenvectors are orthogonal.

---

### Example 2: Finding Eigenvectors of a $3 \times 3$ Matrix

Find the eigenvectors of $A = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{bmatrix}$.

**Step 1**: $A$ is diagonal, so the eigenvalues are the diagonal entries: $\lambda_1 = 2$, $\lambda_2 = 3$, $\lambda_3 = 5$.

**Step 2**: For $\lambda = 2$:

\[
A - 2I = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 3 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]

Equations: $v_2 = 0$, $v_3 = 0$, $v_1$ free. $\mathbf{v} = t \mathbf{e}_1$.

Eigenvector direction: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.

**Step 3**: For $\lambda = 3$: $A - 3I = \operatorname{diag}(-1, 0, 2)$, giving $\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$.

**Step 4**: For $\lambda = 5$: $A - 5I = \operatorname{diag}(-3, -2, 0)$, giving $\mathbf{v}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$.

The eigenvectors are the standard basis vectors, which are orthogonal.

---

### Example 3: Defective Matrix — Repeated Eigenvalue with Insufficient Eigenvectors

Find the eigenvectors of $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$.

**Step 1**: Find eigenvalues. Since $A$ is upper triangular, $\lambda = 1$ (algebraic multiplicity 2).

**Step 2**: Solve $(A - I)\mathbf{v} = \mathbf{0}$:

\[
A - I = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}.
\]

This gives $v_2 = 0$, with $v_1$ free. Parametric: $\mathbf{v} = t \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

**Step 3**: The eigenspace is one-dimensional (geometric multiplicity 1). There is only one linearly independent eigenvector. The matrix is defective and cannot be diagonalised.

---

### Example 4: Complex Eigenvectors

Find the eigenvectors of $A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.

**Step 1**: Eigenvalues: $\det(A - \lambda I) = \lambda^2 + 1 = 0$, so $\lambda = \pm i$.

**Step 2**: For $\lambda = i$:

\[
A - iI = \begin{bmatrix} -i & -1 \\ 1 & -i \end{bmatrix}.
\]

Row-reduce: multiply first row by $i$ (since $-i \cdot i = 1$):
\[
\begin{bmatrix} 1 & -i \\ 1 & -i \end{bmatrix} \to \begin{bmatrix} 1 & -i \\ 0 & 0 \end{bmatrix}.
\]

Equation: $v_1 - i v_2 = 0$, so $v_1 = i v_2$. Parametric: $\mathbf{v} = t \begin{bmatrix} i \\ 1 \end{bmatrix}$.

**Step 3**: For $\lambda = -i$: $A + iI = \begin{bmatrix} i & -1 \\ 1 & i \end{bmatrix}$. Similarly, $v_1 = -i v_2$, giving $\mathbf{v} = t \begin{bmatrix} -i \\ 1 \end{bmatrix}$.

The eigenvectors are complex. No real eigenvector exists because a $90^\circ$ rotation has no direction that is preserved (no real $\lambda$).

---

### Example 5: Gram-Schmidt on an Eigenspace

For $A = \begin{bmatrix} 1 & 2 & 0 \\ 2 & 1 & 0 \\ 0 & 0 & 3 \end{bmatrix}$, find an orthonormal eigenbasis.

**Step 1**: Eigenvalues. Using the $2 \times 2$ block:
\[
\det\begin{bmatrix} 1-\lambda & 2 \\ 2 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 - 4 = \lambda^2 - 2\lambda - 3 = (\lambda - 3)(\lambda + 1) = 0.
\]
So $\lambda = 3$ (multiplicity 2), $\lambda = -1$ (multiplicity 1).

For $\lambda = -1$:
\[
A + I = \begin{bmatrix} 2 & 2 & 0 \\ 2 & 2 & 0 \\ 0 & 0 & 4 \end{bmatrix} \to \begin{bmatrix} 1 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]
$v_1 + v_2 = 0$, $v_3 = 0$. Eigenvector: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}$, normalised: $\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}$.

For $\lambda = 3$:
\[
A - 3I = \begin{bmatrix} -2 & 2 & 0 \\ 2 & -2 & 0 \\ 0 & 0 & 0 \end{bmatrix} \to \begin{bmatrix} 1 & -1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}.
\]
Equation: $v_1 - v_2 = 0$, so $v_1 = v_2$, with $v_3$ free. The eigenspace has dimension 2 with basis:
\[
\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}.
\]

**Step 2**: Apply Gram-Schmidt to obtain an orthonormal basis for $E_3$.

$\mathbf{u}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$, normalise: $\mathbf{e}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$.

$\mathbf{u}_2 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$, already orthogonal to $\mathbf{e}_1$ (dot product = 0), normalise: $\mathbf{e}_2 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$.

**Step 3**: The orthonormal eigenbasis is:
\[
\left\{
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix},
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
\right\}.
\]

## Visual Interpretation

For a $2 \times 2$ real matrix, plot the eigenvectors as arrows from the origin. Under the transformation $A$, each eigenvector arrow is stretched or shrunk but stays on its line. The set of all eigenvectors for a given eigenvalue (plus zero) forms a line (if geometric multiplicity 1) or a plane (if geometric multiplicity 2).

For a symmetric matrix, the eigenvectors are the principal axes of the ellipse formed by mapping the unit circle. These axes are perpendicular. The lengths of the semi-axes are the absolute values of the eigenvalues.

A defective matrix illustrates a geometric degeneracy: there are not enough independent directions preserved by the transformation, so the transformation involves shear-like behaviour in the missing eigendirections.

In 3D, three eigenvectors (if they exist) define a coordinate frame that is aligned with the transformation. The matrix acts as a diagonal scaling in this frame.

## Common Mistakes

1. **Thinking the zero vector is an eigenvector**: By definition, eigenvectors are **non-zero**. The zero vector always satisfies $A \cdot \mathbf{0} = \lambda \cdot \mathbf{0}$ for any $\lambda$, which is trivial and excluded.

2. **Forgetting to check that the eigenvector is non-zero when solving $(A - \lambda I)\mathbf{v} = \mathbf{0}$**: The system always has the trivial solution $\mathbf{v} = \mathbf{0}$. We specifically want the non-trivial solutions.

3. **Confusing eigenvectors of $A$ with eigenvectors of $A^{\mathsf{T}}$**: They are generally different (though eigenvalues are the same).

4. **Assuming eigenvectors are always orthogonal**: This is only guaranteed for symmetric (or normal) matrices. General matrices can have non-orthogonal eigenvectors.

5. **Thinking that repeated eigenvalue always means fewer eigenvectors**: If the eigenvalue's algebraic multiplicity equals its geometric multiplicity, you still get a full set. Example: the identity matrix has $\lambda = 1$ (multiplicity $n$) and $n$ independent eigenvectors.

6. **Forgetting to normalise eigenvectors when required**: In applications like PCA, eigenvectors are typically normalised to unit length. Always check conventions.

7. **Trying to find eigenvectors without first finding eigenvalues**: The eigenvectors for $\lambda$ are defined relative to $\lambda$. You need the eigenvalue first.

8. **Assuming eigenvectors are always real**: If eigenvalues are complex, eigenvectors must also be complex (and occur in conjugate pairs for real matrices).

9. **Using the wrong eigenvalue when solving for eigenvectors**: Each eigenvector must be paired with its correct eigenvalue. Mixing them up gives spurious results.

10. **Believing that every matrix has $n$ linearly independent eigenvectors**: Only diagonalisable matrices have a full set. Defective matrices lack a complete eigenbasis.

## Interview Questions

### Beginner

1. **Q**: What is an eigenvector?
   **A**: A non-zero vector $\mathbf{v}$ such that $A\mathbf{v} = \lambda \mathbf{v}$ for some scalar $\lambda$ (the eigenvalue). The eigenvector's direction is preserved under the transformation $A$.

2. **Q**: How do you find an eigenvector once you know the eigenvalue?
   **A**: Solve the homogeneous system $(A - \lambda I)\mathbf{v} = \mathbf{0}$. The non-zero solutions are the eigenvectors.

3. **Q**: What is the eigenspace?
   **A**: The eigenspace $E_\lambda$ is the set of all eigenvectors for $\lambda$ together with $\mathbf{0}$, i.e., $\ker(A - \lambda I)$. It is a subspace.

4. **Q**: If $\mathbf{v}$ is an eigenvector of $A$, is $2\mathbf{v}$ also an eigenvector?
   **A**: Yes. $A(2\mathbf{v}) = 2A\mathbf{v} = 2\lambda \mathbf{v} = \lambda (2\mathbf{v})$. Any non-zero scalar multiple of an eigenvector is also an eigenvector.

5. **Q**: Can a matrix have a non-zero eigenvector for $\lambda = 0$?
   **A**: Yes. An eigenvector for $\lambda = 0$ satisfies $A\mathbf{v} = \mathbf{0}$, meaning $\mathbf{v}$ is in the nullspace of $A$. This happens when $A$ is singular.

### Intermediate

1. **Q**: Prove that eigenvectors corresponding to distinct eigenvalues are linearly independent.
   **A**: Suppose $\mathbf{v}_1, \dots, \mathbf{v}_k$ are eigenvectors with distinct eigenvalues $\lambda_1, \dots, \lambda_k$. Assume $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k = \mathbf{0}$. Apply $A$: $c_1\lambda_1\mathbf{v}_1 + \dots + c_k\lambda_k\mathbf{v}_k = \mathbf{0}$. Subtract $\lambda_k$ times the first equation: $c_1(\lambda_1 - \lambda_k)\mathbf{v}_1 + \dots + c_{k-1}(\lambda_{k-1} - \lambda_k)\mathbf{v}_{k-1} = \mathbf{0}$. By induction, since $\lambda_i \neq \lambda_k$ for $i < k$, all coefficients must be zero. Hence the eigenvectors are independent.

2. **Q**: For a real symmetric matrix, why are eigenvectors corresponding to distinct eigenvalues orthogonal?
   **A**: Let $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$. Then $\lambda_1 \mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2 = (A\mathbf{v}_1)^{\mathsf{T}} \mathbf{v}_2 = \mathbf{v}_1^{\mathsf{T}} A^{\mathsf{T}} \mathbf{v}_2 = \mathbf{v}_1^{\mathsf{T}} A \mathbf{v}_2 = \lambda_2 \mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2$. So $(\lambda_1 - \lambda_2)(\mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2) = 0$. Since $\lambda_1 \neq \lambda_2$, we have $\mathbf{v}_1^{\mathsf{T}} \mathbf{v}_2 = 0$.

3. **Q**: What is a defective matrix?
   **A**: A matrix whose geometric multiplicity is less than the algebraic multiplicity for at least one eigenvalue. Such a matrix does not have $n$ linearly independent eigenvectors and cannot be diagonalised.

4. **Q**: What are left eigenvectors and how do they relate to right eigenvectors?
   **A**: A left eigenvector $\mathbf{w}^{\mathsf{T}}$ satisfies $\mathbf{w}^{\mathsf{T}} A = \lambda \mathbf{w}^{\mathsf{T}}$. Left eigenvectors of $A$ are right eigenvectors of $A^{\mathsf{T}}$. For symmetric matrices, left and right eigenvectors coincide.

5. **Q**: In PCA, why are the eigenvectors of the covariance matrix called "principal components"?
   **A**: They are the directions of maximum variance. The first eigenvector maximises $\operatorname{Var}(X\mathbf{v})$ subject to $\|\mathbf{v}\| = 1$. Each subsequent eigenvector maximises variance under orthogonality to previous ones. The eigenvectors form the principal axes of the data ellipsoid.

### Advanced

1. **Q**: Explain how eigenvectors are used in PageRank and why the power iteration converges to the dominant eigenvector.
   **A**: The Google matrix $G = \alpha S + (1-\alpha)\frac{1}{n} \mathbf{1}\mathbf{1}^{\mathsf{T}}$ is a stochastic matrix with spectral radius $1$. Its dominant eigenvector (the PageRank vector) satisfies $G\mathbf{v} = \mathbf{v}$. Power iteration $\mathbf{v}_{k+1} = G\mathbf{v}_k$ converges to this eigenvector because all other eigenvalues have magnitude $< \alpha < 1$, so the component along any other eigenvector decays geometrically. The damping factor $\alpha \approx 0.85$ ensures a unique dominant eigenvector and fast convergence.

2. **Q**: In spectral clustering, how do the eigenvectors of the graph Laplacian reveal cluster structure?
   **A**: For a graph with $k$ connected components, the Laplacian $L$ has $k$ eigenvectors with eigenvalue $0$, each being the indicator vector of a component. For a graph with well-separated clusters (near-disconnected), the first $k$ eigenvectors of $L$ are approximately piecewise-constant on the clusters, and the rows of $U \in \mathbb{R}^{N \times k}$ form $k$ tight clusters in $\mathbb{R}^{k}$. $k$-means on these rows recovers the clusters. The eigenvalue gap $\lambda_{k+1} - \lambda_k$ indicates how well-separated the clusters are.

3. **Q**: For a normal matrix $A$ ($AA^{\mathsf{T}} = A^{\mathsf{T}} A$), prove that eigenvectors corresponding to distinct eigenvalues are orthogonal.
   **A**: Let $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$. For normal matrices, $A$ is unitarily diagonalisable: $A = U \Lambda U^*$. Then $\mathbf{v}_1 = U \mathbf{e}_i$ and $\mathbf{v}_2 = U \mathbf{e}_j$ for some $i \neq j$, so $\mathbf{v}_1^* \mathbf{v}_2 = \mathbf{e}_i^{\mathsf{T}} \mathbf{e}_j = 0$. Alternatively, use the fact that $A - \lambda I$ is normal and preserve orthogonality of eigenspaces. This includes symmetric, skew-symmetric, orthogonal, and Hermitian matrices as special cases.

## Practice Problems

### Easy

1. Find an eigenvector of $A = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}$ for $\lambda = 3$.

2. Find a basis for the eigenspace of $A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ corresponding to $\lambda = 2$.

3. For $A = \begin{bmatrix} 1 & 2 \\ 0 & 3 \end{bmatrix}$, find eigenvectors for each eigenvalue.

4. Is $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ an eigenvector of $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$? If so, find the eigenvalue.

5. Let $\mathbf{v} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ be an eigenvector of $A = \begin{bmatrix} 3 & a \\ 4 & b \end{bmatrix}$ with eigenvalue $2$. Find $a$ and $b$.

### Medium

1. Find all eigenvectors of $A = \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}$ (including normalisation).

2. For $A = \begin{bmatrix} 4 & -5 \\ 2 & -3 \end{bmatrix}$, find the eigenvalues and a basis of eigenvectors.

3. Find an orthonormal basis of eigenvectors for $A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$.

4. Show that if $\mathbf{v}$ is an eigenvector of $A$ with eigenvalue $\lambda$, then $\mathbf{v}$ is also an eigenvector of $A^3$ with eigenvalue $\lambda^3$.

5. Determine whether $A = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix}$ has a full set of eigenvectors. If not, find the eigenvectors it does have.

### Hard

1. Let $A$ be symmetric and positive semi-definite. Prove that the eigenvectors corresponding to non-zero eigenvalues span $\operatorname{im}(A)$ and that $\operatorname{im}(A) \perp \ker(A)$.

2. Given $A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$, find all eigenvalues, their eigenspaces, and construct an orthonormal eigenbasis. Use this to verify the spectral decomposition $A = \lambda_1 \mathbf{v}_1 \mathbf{v}_1^{\mathsf{T}} + \lambda_2 \mathbf{v}_2 \mathbf{v}_2^{\mathsf{T}}$.

3. Prove that if $\mathbf{v}$ is an eigenvector of $A$ with eigenvalue $\lambda$, and also an eigenvector of $B$ with eigenvalue $\mu$, then $\mathbf{v}$ is an eigenvector of $AB$ with eigenvalue $\lambda\mu$. Is the converse true? That is, if $\mathbf{v}$ is an eigenvector of $AB$, must it be an eigenvector of $A$ and $B$ individually?

## Solutions

### Easy Solutions

**1.** For $\lambda = 3$, solve $(A - 3I)\mathbf{v} = \mathbf{0}$:
\[
\begin{bmatrix} 0 & 0 \\ 0 & -5 \end{bmatrix} \mathbf{v} = \mathbf{0}.
\]
This gives $v_2 = 0$, $v_1$ free. One eigenvector: $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

**2.** $A - 2I = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$, so every non-zero vector in $\mathbb{R}^2$ is an eigenvector. Basis: $\{\mathbf{e}_1, \mathbf{e}_2\}$.

**3.** Eigenvalues: the matrix is upper triangular, so $\lambda_1 = 1$, $\lambda_2 = 3$.

For $\lambda = 1$:
\[
A - I = \begin{bmatrix} 0 & 2 \\ 0 & 2 \end{bmatrix} \to \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}.
\]
Equation: $v_2 = 0$, $v_1$ free. Eigenvector: $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

For $\lambda = 3$:
\[
A - 3I = \begin{bmatrix} -2 & 2 \\ 0 & 0 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]
Equation: $v_1 - v_2 = 0$. Eigenvector: $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

**4.** Compute $A\mathbf{v} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 1 \end{bmatrix}$. Yes, $\mathbf{v}$ is an eigenvector with eigenvalue $\lambda = 3$.

**5.** Since $A\mathbf{v} = \lambda \mathbf{v}$:
\[
\begin{bmatrix} 3 & a \\ 4 & b \end{bmatrix} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 3 - a \\ 4 - b \end{bmatrix} = 2 \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \\ -2 \end{bmatrix}.
\]
So $3 - a = 2 \implies a = 1$, and $4 - b = -2 \implies b = 6$.

### Medium Solutions

**1.** Eigenvalues:
\[
\det\begin{bmatrix} 1-\lambda & 3 \\ 3 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 - 9 = \lambda^2 - 2\lambda - 8 = (\lambda - 4)(\lambda + 2) = 0.
\]
So $\lambda_1 = 4$, $\lambda_2 = -2$.

For $\lambda = 4$:
\[
A - 4I = \begin{bmatrix} -3 & 3 \\ 3 & -3 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ (normalised).

For $\lambda = -2$:
\[
A + 2I = \begin{bmatrix} 3 & 3 \\ 3 & 3 \end{bmatrix} \to \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ (normalised).

**2.** Eigenvalues:
\[
\det\begin{bmatrix} 4-\lambda & -5 \\ 2 & -3-\lambda \end{bmatrix} = (4-\lambda)(-3-\lambda) + 10 = \lambda^2 - \lambda - 12 + 10 = \lambda^2 - \lambda - 2 = (\lambda - 2)(\lambda + 1) = 0.
\]
$\lambda_1 = 2$, $\lambda_2 = -1$.

For $\lambda = 2$:
\[
A - 2I = \begin{bmatrix} 2 & -5 \\ 2 & -5 \end{bmatrix} \to \begin{bmatrix} 1 & -\tfrac52 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_1 = \begin{bmatrix} 5 \\ 2 \end{bmatrix}$.

For $\lambda = -1$:
\[
A + I = \begin{bmatrix} 5 & -5 \\ 2 & -2 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

**3.** Eigenvalues:
\[
\det\begin{bmatrix} 1-\lambda & 2 \\ 2 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 - 4 = \lambda^2 - 2\lambda - 3 = (\lambda - 3)(\lambda + 1) = 0.
\]
$\lambda_1 = 3$, $\lambda_2 = -1$.

For $\lambda = 3$:
\[
A - 3I = \begin{bmatrix} -2 & 2 \\ 2 & -2 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

For $\lambda = -1$:
\[
A + I = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} \to \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

Orthonormal eigenbasis: $\{\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}, \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}\}$.

**4.** If $A\mathbf{v} = \lambda \mathbf{v}$, then $A^3 \mathbf{v} = A(A(A\mathbf{v})) = A(A(\lambda \mathbf{v})) = A(\lambda^2 \mathbf{v}) = \lambda^3 \mathbf{v}$. So $\mathbf{v}$ is also an eigenvector of $A^3$ with eigenvalue $\lambda^3$.

**5.** $A$ is upper triangular, so $\lambda = 1$ (algebraic multiplicity 3). Solve $(A - I)\mathbf{v} = \mathbf{0}$:
\[
A - I = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]
Equations: $v_2 = 0$, $v_3 = 0$, $v_1$ free. Only one eigenvector direction: $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.

The geometric multiplicity is 1, which is less than the algebraic multiplicity of 3. The matrix is defective — it does NOT have a full set of eigenvectors.

### Hard Solutions

**1.** Since $A$ is symmetric, it has an orthonormal eigenbasis $\{\mathbf{v}_1, \dots, \mathbf{v}_n\}$. Let $\lambda_1 \geq \dots \geq \lambda_r > 0$ be the non-zero eigenvalues and $\lambda_{r+1} = \dots = \lambda_n = 0$.

For any $\mathbf{x} \in \mathbb{R}^n$, write $\mathbf{x} = \sum_{i=1}^n c_i \mathbf{v}_i$. Then $A\mathbf{x} = \sum_{i=1}^r c_i \lambda_i \mathbf{v}_i$, so $\operatorname{im}(A) = \operatorname{span}\{\mathbf{v}_1, \dots, \mathbf{v}_r\}$.

For $\mathbf{x} \in \ker(A)$: $A\mathbf{x} = \sum_{i=1}^r c_i \lambda_i \mathbf{v}_i = \mathbf{0}$, which implies $c_1 = \dots = c_r = 0$, so $\mathbf{x} \in \operatorname{span}\{\mathbf{v}_{r+1}, \dots, \mathbf{v}_n\}$.

Thus $\operatorname{im}(A) = \operatorname{span}\{\mathbf{v}_1, \dots, \mathbf{v}_r\}$ and $\ker(A) = \operatorname{span}\{\mathbf{v}_{r+1}, \dots, \mathbf{v}_n\}$, which are orthogonal. Therefore $\mathbb{R}^n = \operatorname{im}(A) \oplus \ker(A)$ (orthogonal decomposition).

**2.** Eigenvalues:
\[
\det\begin{bmatrix} 1-\lambda & 1 \\ 1 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 - 1 = \lambda^2 - 2\lambda = \lambda(\lambda - 2) = 0.
\]
$\lambda_1 = 0$, $\lambda_2 = 2$.

For $\lambda = 0$:
\[
A - 0I = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \to \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

For $\lambda = 2$:
\[
A - 2I = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

Spectral decomposition:
\[
A = 2 \cdot \left(\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right)\left(\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \end{bmatrix}\right) + 0 \cdot \left(\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}\right)\left(\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1 \end{bmatrix}\right).
\]
So $A = 2 \cdot \frac12 \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$. ✓

**3.** $AB\mathbf{v} = A(\mu \mathbf{v}) = \mu A\mathbf{v} = \mu \lambda \mathbf{v} = \lambda \mu \mathbf{v}$. So yes, $\mathbf{v}$ is an eigenvector of $AB$ with eigenvalue $\lambda\mu$.

The converse is false. Counterexample: Let $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ and $B = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$. Then $AB = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$, so every vector is an eigenvector of $AB$ with eigenvalue $0$. But $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ is not an eigenvector of $A$ ($A\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \neq \lambda \mathbf{v}$) or $B$.

## Related Concepts

- **Eigenvalue (MATH-039)**: The scalar $\lambda$ paired with an eigenvector; the root of the characteristic polynomial.
- **Eigenspace**: $\ker(A - \lambda I)$, the set of all eigenvectors for $\lambda$ plus $\mathbf{0}$.
- **Characteristic Polynomial**: $\det(A - \lambda I)$, used to find eigenvalues whose eigenvectors are then computed.
- **Diagonalization (MATH-041)**: Using eigenvectors as columns of $P$ to factor $A = PDP^{-1}$.
- **Spectral Theorem**: Symmetric matrices have an orthonormal eigenbasis.
- **Gram-Schmidt Process**: Orthonormalising an eigenbasis for symmetric matrices with repeated eigenvalues.
- **Principal Component Analysis**: Using eigenvectors of the covariance matrix for dimensionality reduction.
- **Graph Laplacian**: The eigenvectors of $L$ reveal connected components and clusters.
- **Generalised Eigenvector**: For defective matrices, generalised eigenvectors extend the eigenvector concept to achieve Jordan form.

## Next Concepts

- **Diagonalization (MATH-041)**: Using eigenvectors to decompose $A$ into $PDP^{-1}$.
- **Generalised Eigenvectors and Jordan Form**: Handling defective matrices.
- **Singular Value Decomposition**: Left singular vectors are eigenvectors of $AA^{\mathsf{T}}$; right singular vectors are eigenvectors of $A^{\mathsf{T}}A$.
- **Spectral Clustering**: Applying Laplacian eigenvectors to community detection.
- **Power Iteration**: Algorithm for finding the dominant eigenvector.
- **QR Algorithm**: Numerically stable method for computing all eigenvalues and eigenvectors.

## Summary

Eigenvectors are the non-zero vectors that satisfy $A\mathbf{v} = \lambda \mathbf{v}$, representing directions that are preserved (only scaled) by the linear transformation. For each eigenvalue $\lambda$, the set of eigenvectors forms a subspace $E_\lambda = \ker(A - \lambda I)$ called the eigenspace. Eigenvectors corresponding to distinct eigenvalues are linearly independent. For real symmetric matrices, eigenvectors for distinct eigenvalues are orthogonal, and an orthonormal eigenbasis exists. A matrix with $n$ linearly independent eigenvectors is diagonalisable; otherwise, it is defective. Eigenvectors are fundamental to PCA (principal directions of maximum variance), spectral clustering (Laplacian eigenvectors reveal clusters), PageRank (dominant eigenvector of the Google matrix), and LDA (Fisher discriminant directions). The geometric multiplicity of an eigenvalue is the dimension of its eigenspace.

## Key Takeaways

- Eigenvectors are non-zero vectors satisfying $A\mathbf{v} = \lambda \mathbf{v}$.
- Find eigenvectors by solving $(A - \lambda I)\mathbf{v} = \mathbf{0}$ for each eigenvalue.
- The eigenspace $E_\lambda = \ker(A - \lambda I)$ is a subspace containing all eigenvectors for $\lambda$.
- Eigenvectors for distinct eigenvalues are linearly independent.
- For symmetric matrices, eigenvectors for distinct eigenvalues are orthogonal.
- A matrix is diagonalisable iff it has $n$ linearly independent eigenvectors.
- In PCA, the eigenvectors of $X^{\mathsf{T}} X$ give the principal directions of maximum variance.
- In spectral clustering, the eigenvectors of the graph Laplacian reveal cluster structure.
- In PageRank, the dominant eigenvector of the Google matrix ranks web pages by importance.
