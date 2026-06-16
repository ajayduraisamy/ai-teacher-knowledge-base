# Concept: Eigenvalue

## Concept ID

MATH-039

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Define eigenvalues as scalars $\lambda$ satisfying $A\mathbf{v} = \lambda \mathbf{v}$ for a given square matrix $A$.
- Derive and solve the characteristic polynomial $\det(A - \lambda I) = 0$ to find eigenvalues.
- Interpret the spectrum of a matrix and distinguish algebraic from geometric multiplicity.
- Relate the trace and determinant of a matrix to the sum and product of its eigenvalues.
- Compute eigenvalues for $2 \times 2$ and $3 \times 3$ matrices analytically.
- Apply eigenvalue analysis to understand the behaviour of linear dynamical systems, PCA variance decomposition, and optimisation convergence rates.

## Prerequisites

- Matrix operations, multiplication, and transposition.
- Determinants and their computation (MATH-027).
- Solving polynomial equations (quadratic and cubic).
- Basic vector spaces and linear transformations (MATH-031, MATH-036).
- Complex numbers (MATH-009) — eigenvalues can be complex even for real matrices.

## Definition

Let $A \in \mathbb{C}^{n \times n}$ be a square matrix. A scalar $\lambda \in \mathbb{C}$ is called an **eigenvalue** of $A$ if there exists a non-zero vector $\mathbf{v} \in \mathbb{C}^{n}$ such that:

\[
A \mathbf{v} = \lambda \mathbf{v}.
\]

The vector $\mathbf{v}$ is called the corresponding **eigenvector**. The eigenvalue equation says that applying $A$ to $\mathbf{v}$ simply scales $\mathbf{v}$ by $\lambda$ — the direction of $\mathbf{v}$ is preserved (or reversed if $\lambda < 0$, or rotated if $\lambda$ is complex).

Rearranging the eigenvalue equation gives the homogeneous linear system:

\[
(A - \lambda I) \mathbf{v} = \mathbf{0}.
\]

For a non-zero solution $\mathbf{v}$ to exist, the matrix $A - \lambda I$ must be singular — its determinant must be zero. This leads to the **characteristic equation**:

\[
\det(A - \lambda I) = 0.
\]

Expanding the determinant yields a degree-$n$ polynomial in $\lambda$ called the **characteristic polynomial** $p(\lambda) = \det(A - \lambda I)$. The eigenvalues of $A$ are precisely the roots of this polynomial.

## Intuition

Imagine a linear transformation $T : \mathbb{R}^{n} \to \mathbb{R}^{n}$ represented by matrix $A$. Most vectors change direction when $T$ is applied. Eigenvalues correspond to the special scalars that describe how much certain "special vectors" (eigenvectors) are stretched or squashed along directions that remain invariant. If you think of a matrix as a function that moves points around, eigenvalues tell you the scale factors along the directions that do not rotate.

Geometrically, for a $2 \times 2$ real matrix with real eigenvalues, each eigenvalue $\lambda_i$ gives the factor by which the transformation stretches or compresses space along the corresponding eigenvector direction. A negative eigenvalue indicates a reflection. An eigenvalue with magnitude greater than $1$ means expansion; less than $1$ means contraction. Complex eigenvalues (appearing as conjugate pairs) correspond to rotations in the plane.

## Why This Concept Matters

Eigenvalues are among the most powerful descriptors of a linear transformation. They determine:

- **Stability**: In dynamical systems $\mathbf{x}_{k+1} = A \mathbf{x}_k$, the system is stable iff all eigenvalues have magnitude $< 1$.
- **Invertibility**: $A$ is invertible iff $0$ is not an eigenvalue (i.e., $\det(A) \neq 0$).
- **Spectral decomposition**: Symmetric matrices can be orthogonally diagonalised using their eigenvalues.
- **Energy**: The eigenvalues of the covariance matrix quantify the variance captured along each principal component.
- **Conditioning**: The ratio of the largest to smallest eigenvalue magnitude (the condition number) controls the convergence rate of iterative optimisation methods.

Without eigenvalues, fields like quantum mechanics (energy levels are eigenvalues), structural engineering (resonant frequencies), and machine learning (PCA, spectral clustering, convergence analysis) would lose their mathematical foundation.

## Historical Background

The concept of eigenvalues emerged gradually over two centuries. **Leonhard Euler** (1707–1783) studied principal axes of rotation, effectively computing eigenvalues of inertia matrices. **Joseph-Louis Lagrange** (1736–1813) used eigenvalue-like ideas in celestial mechanics. The term "eigenvalue" comes from the German word *eigen* meaning "own" or "characteristic"; **David Hilbert** (1862–1943) popularised this term in the early 20th century.

**Augustin-Louis Cauchy** (1789–1857) established the characteristic equation $\det(A - \lambda I) = 0$ and showed that eigenvalues of symmetric matrices are real. **Camille Jordan** (1838–1922) developed the Jordan normal form, revealing the structure of non-diagonalisable matrices. **James Joseph Sylvester** (1814–1897) coined the term "matrix" and studied the inertia of quadratic forms. **Charles Hermite** (1822–1901) extended the eigenvalue theory to complex matrices with the concept now known as Hermitian matrices.

The spectral theorem — that every real symmetric matrix has a complete set of orthogonal eigenvectors — was developed through the work of **Hermann Weyl** (1885–1955) and **John von Neumann** (1903–1957) in the context of quantum mechanics and operator theory.

## Real World Examples

### 1. Structural Engineering — Resonant Frequencies

A bridge or building has natural resonant frequencies at which it vibrates when subjected to forces like wind or earthquakes. These frequencies are the square roots of the eigenvalues of the stiffness matrix divided by the mass matrix. Engineers design structures so that the lowest eigenvalues are far from the frequencies of external excitations to avoid catastrophic resonance.

### 2. Google PageRank

The PageRank algorithm computes the stationary distribution of a random walk on the web graph. The PageRank vector is the eigenvector of the Google matrix corresponding to eigenvalue $\lambda = 1$. The power iteration method (repeated multiplication by the Google matrix) converges because the dominant eigenvalue is $1$ and all other eigenvalues have smaller magnitude.

### 3. Vibration Analysis — Modal Decomposition

In a multi-degree-of-freedom mechanical system, the equations of motion involve a mass matrix $M$ and a stiffness matrix $K$. Solving the generalised eigenvalue problem $K \mathbf{v} = \lambda M \mathbf{v}$ gives the natural frequencies $\sqrt{\lambda}$ and mode shapes $\mathbf{v}$ of the system. This is used to simulate how structures respond to dynamic loads.

### 4. Image Compression — PCA

In Principal Component Analysis (PCA), the eigenvalues of the covariance matrix $X^{\mathsf{T}} X$ represent the variance captured by each principal component. By retaining only the top $k$ components (those with the largest eigenvalues), one can compress images while preserving most of the visual information.

### 5. Population Dynamics

In a Leslie matrix model of population growth, the dominant eigenvalue determines the long-term growth rate of the population. The corresponding eigenvector gives the stable age distribution that the population converges to over time.

## AI/ML Relevance

### 1. Principal Component Analysis (PCA)

Given a data matrix $X \in \mathbb{R}^{N \times d}$ (centred), the covariance matrix is $S = \frac{1}{N} X^{\mathsf{T}} X \in \mathbb{R}^{d \times d}$. The eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \geq 0$ of $S$ represent the variance explained by each principal component. The proportion of total variance captured by the first $k$ components is:

\[
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}.
\]

Selecting $k$ such that this ratio exceeds a threshold (e.g., $0.95$) is a standard dimensionality reduction technique. In facial recognition (eigenfaces) and image compression, the eigenvalues guide which components to retain.

### 2. Convergence of Gradient Descent

For a quadratic objective $f(\mathbf{x}) = \frac12 \mathbf{x}^{\mathsf{T}} H \mathbf{x} - \mathbf{b}^{\mathsf{T}} \mathbf{x}$ with positive definite Hessian $H$, gradient descent with step size $\alpha$ converges at a rate governed by the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ of $H$. The optimal step size is $\alpha^* = 2 / (\lambda_{\max} + \lambda_{\min})$, and the convergence factor is:

\[
\rho = \frac{\kappa - 1}{\kappa + 1}.
\]

A large condition number (ill-conditioned problem) leads to slow convergence. This is why preconditioning — transforming the problem to have eigenvalues closer together — is critical in large-scale optimisation.

### 3. Spectral Radius and Stability in Neural Networks

The **spectral radius** $\rho(A) = \max_i |\lambda_i|$ of the recurrent weight matrix $A$ in an RNN controls whether the hidden state dynamics are stable. If $\rho(A) > 1$, gradients tend to explode; if $\rho(A) < 1$, gradients vanish. This insight motivated techniques like spectral normalisation (constraining the spectral radius to $1$) and orthogonal initialisation (setting all eigenvalues to have magnitude $1$).

### 4. Covariance Matrices in Gaussian Processes

The kernel matrix $K$ in a Gaussian process must be positive semi-definite (all eigenvalues $\geq 0$). The eigenvalues of $K$ determine the smoothness and length-scale of the GP posterior. The log-marginal likelihood involves $\log \det(K) = \sum_i \log \lambda_i$, which appears in the model selection objective.

### 5. Manifold Learning — Diffusion Maps

Diffusion maps construct a Markov matrix $P$ whose eigenvalues decay rapidly. The embedding coordinates are scaled eigenvectors of $P$, where the eigenvalue $\lambda_i$ indicates the importance of the $i$-th coordinate. The spectral gap $\lambda_k - \lambda_{k+1}$ helps determine the intrinsic dimensionality of the data manifold.

### 6. Graph Neural Networks — Spectral Graph Theory

The eigenvalues of the graph Laplacian $L = D - A$ encode structural properties of the graph. In spectral graph convolutional networks (ChebNet, GCN), filters are defined as polynomials of the eigenvalues, enabling convolutions on irregular graphs.

### 7. Normalising Flows — Jacobian Eigenvalues

In continuous normalising flows (Neural ODEs), the instantaneous change-of-variables formula involves the trace (sum of eigenvalues) of the Jacobian. The stability of the flow is analysed through the eigenvalues of the Jacobian matrix of the vector field.

## Mathematical Explanation

### Derivation from the Eigenvalue Equation

Starting from $A\mathbf{v} = \lambda \mathbf{v}$, rewrite as $A\mathbf{v} - \lambda I \mathbf{v} = \mathbf{0}$, giving $(A - \lambda I)\mathbf{v} = \mathbf{0}$. For this homogeneous system to have a non-trivial solution ($\mathbf{v} \neq \mathbf{0}$), the matrix $A - \lambda I$ must be singular:

\[
\det(A - \lambda I) = 0.
\]

This determinant expands to the **characteristic polynomial**:

\[
p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + c_{n-1} \lambda^{n-1} + \dots + c_1 \lambda + c_0.
\]

The roots of $p(\lambda) = 0$ are the eigenvalues of $A$.

### Algebraic and Geometric Multiplicity

The **algebraic multiplicity** of an eigenvalue $\lambda$ is its multiplicity as a root of the characteristic polynomial. The **geometric multiplicity** is the dimension of the eigenspace $\ker(A - \lambda I)$, i.e., the number of linearly independent eigenvectors associated with $\lambda$. Always:

\[
1 \leq \text{geometric multiplicity} \leq \text{algebraic multiplicity}.
\]

If the geometric multiplicity is strictly less than the algebraic multiplicity, the matrix is **defective** (non-diagonalisable).

### The Spectrum

The set of all eigenvalues of $A$ is called the **spectrum** of $A$, denoted $\sigma(A)$. The **spectral radius** is $\rho(A) = \max\{|\lambda| : \lambda \in \sigma(A)\}$.

### Trace and Determinant Relations

For any $A \in \mathbb{C}^{n \times n}$:

\[
\operatorname{tr}(A) = \sum_{i=1}^{n} \lambda_i, \qquad
\det(A) = \prod_{i=1}^{n} \lambda_i.
\]

These follow from expanding $p(\lambda) = \det(A - \lambda I) = \prod_{i=1}^n (\lambda_i - \lambda)$ and comparing coefficients.

### Eigenvalues of Special Matrices

- **Symmetric real matrices**: All eigenvalues are real. The eigenvectors can be chosen to be orthogonal.
- **Positive definite matrices**: All eigenvalues are positive.
- **Orthogonal matrices**: All eigenvalues have magnitude $1$ (they lie on the unit circle).
- **Skew-symmetric matrices**: All eigenvalues are purely imaginary or zero.
- **Triangular matrices**: The eigenvalues are the diagonal entries.
- **Idempotent matrices** ($A^2 = A$): Eigenvalues are only $0$ or $1$.

## Formula(s)

\[
A \mathbf{v} = \lambda \mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}
\]

\[
\det(A - \lambda I) = 0
\]

\[
p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + c_{n-1} \lambda^{n-1} + \dots + c_0
\]

\[
\operatorname{tr}(A) = \sum_{i=1}^n \lambda_i
\]

\[
\det(A) = \prod_{i=1}^n \lambda_i
\]

\[
\rho(A) = \max_i |\lambda_i|
\]

## Properties

1. **Similarity invariance**: If $B = P^{-1} A P$, then $A$ and $B$ have the same eigenvalues. The characteristic polynomial is invariant under similarity transformations.

2. **Eigenvalues of $A^{-1}$**: If $A$ is invertible, $\lambda$ is an eigenvalue of $A$ iff $1/\lambda$ is an eigenvalue of $A^{-1}$.

3. **Eigenvalues of $A^k$**: If $\lambda$ is an eigenvalue of $A$, then $\lambda^k$ is an eigenvalue of $A^k$ for any integer $k \geq 0$.

4. **Eigenvalues of $p(A)$**: If $\lambda$ is an eigenvalue of $A$ and $p$ is a polynomial, then $p(\lambda)$ is an eigenvalue of $p(A)$.

5. **Eigenvalues of $A^{\mathsf{T}}$**: $A$ and $A^{\mathsf{T}}$ have the same eigenvalues (same characteristic polynomial).

6. **Real eigenvalues for symmetric $A$**: All eigenvalues of a real symmetric matrix are real.

7. **Non-negative eigenvalues for PSD**: A real symmetric matrix is positive semi-definite iff all its eigenvalues are $\geq 0$.

8. **Gershgorin circle theorem**: Every eigenvalue of $A$ lies in at least one Gershgorin disc $D_i = \{z \in \mathbb{C} : |z - a_{ii}| \leq \sum_{j \neq i} |a_{ij}|\}$.

9. **Spectral radius bound**: $\rho(A) \leq \|A\|$ for any matrix norm induced by a vector norm.

10. **Eigenvalues of a block triangular matrix**: The eigenvalues are the union of the eigenvalues of the diagonal blocks.

## Step-by-Step Worked Examples

### Example 1: $2 \times 2$ Matrix with Real Distinct Eigenvalues

Find the eigenvalues of $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$.

**Step 1**: Write the characteristic equation:

\[
\det(A - \lambda I) = \det\begin{bmatrix} 4 - \lambda & 1 \\ 2 & 3 - \lambda \end{bmatrix} = 0.
\]

**Step 2**: Compute the determinant:

\[
(4 - \lambda)(3 - \lambda) - (1)(2) = 0.
\]

**Step 3**: Expand:

\[
12 - 4\lambda - 3\lambda + \lambda^2 - 2 = \lambda^2 - 7\lambda + 10 = 0.
\]

**Step 4**: Solve the quadratic:

\[
\lambda^2 - 7\lambda + 10 = (\lambda - 2)(\lambda - 5) = 0.
\]

**Step 5**: The eigenvalues are $\lambda_1 = 2$ and $\lambda_2 = 5$.

**Verification**: $\operatorname{tr}(A) = 4 + 3 = 7 = 2 + 5$. $\det(A) = 4 \cdot 3 - 1 \cdot 2 = 12 - 2 = 10 = 2 \cdot 5$. ✓

---

### Example 2: $2 \times 2$ Matrix with Complex Eigenvalues

Find the eigenvalues of $A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ (the $90^\circ$ rotation matrix).

**Step 1**: Characteristic equation:

\[
\det\begin{bmatrix} -\lambda & -1 \\ 1 & -\lambda \end{bmatrix} = \lambda^2 + 1 = 0.
\]

**Step 2**: Solve:

\[
\lambda = \pm i.
\]

The eigenvalues are $\lambda_1 = i$ and $\lambda_2 = -i$. These are a complex conjugate pair.

**Verification**: $\operatorname{tr}(A) = 0 = i + (-i)$. $\det(A) = 1 = i \cdot (-i) = 1$. ✓

---

### Example 3: $3 \times 3$ Matrix

Find the eigenvalues of $A = \begin{bmatrix} 3 & -1 & 0 \\ -1 & 3 & 0 \\ 0 & 0 & 2 \end{bmatrix}$.

**Step 1**: Write the characteristic equation:

\[
\det(A - \lambda I) = \det\begin{bmatrix} 3 - \lambda & -1 & 0 \\ -1 & 3 - \lambda & 0 \\ 0 & 0 & 2 - \lambda \end{bmatrix} = 0.
\]

**Step 2**: Expand along the third row (contains two zeros):

\[
(2 - \lambda) \cdot \det\begin{bmatrix} 3 - \lambda & -1 \\ -1 & 3 - \lambda \end{bmatrix} = 0.
\]

**Step 3**: Compute the $2 \times 2$ determinant:

\[
(2 - \lambda)\big[(3 - \lambda)^2 - (-1)(-1)\big] = (2 - \lambda)\big[(3 - \lambda)^2 - 1\big] = 0.
\]

**Step 4**: Expand:

\[
(2 - \lambda)(9 - 6\lambda + \lambda^2 - 1) = (2 - \lambda)(\lambda^2 - 6\lambda + 8) = 0.
\]

**Step 5**: Factor the quadratic:

\[
\lambda^2 - 6\lambda + 8 = (\lambda - 2)(\lambda - 4) = 0.
\]

**Step 6**: The full factorisation:

\[
(2 - \lambda)(\lambda - 2)(\lambda - 4) = 0.
\]

So $\lambda = 2$ (algebraic multiplicity 2) and $\lambda = 4$ (algebraic multiplicity 1).

**Verification**: $\operatorname{tr}(A) = 3 + 3 + 2 = 8 = 2 + 2 + 4$. $\det(A)$: expand original determinant with $\lambda=0$: $\det(A) = \begin{vmatrix} 3 & -1 & 0 \\ -1 & 3 & 0 \\ 0 & 0 & 2 \end{vmatrix} = 2(9-1) = 16 = 2 \cdot 2 \cdot 4$. ✓

---

### Example 4: Repeated Eigenvalue (Algebraic vs Geometric Multiplicity)

Find the eigenvalues of $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and determine their multiplicities.

**Step 1**: Characteristic equation:

\[
\det\begin{bmatrix} 1 - \lambda & 1 \\ 0 & 1 - \lambda \end{bmatrix} = (1 - \lambda)^2 = 0.
\]

**Step 2**: $\lambda = 1$ with algebraic multiplicity 2.

**Step 3**: Find the eigenspace: solve $(A - I)\mathbf{v} = \mathbf{0}$.

\[
A - I = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}.
\]

This gives $v_2 = 0$, with $v_1$ free. So $\mathbf{v} = \begin{bmatrix} t \\ 0 \end{bmatrix}$, a one-dimensional eigenspace.

**Step 4**: Geometric multiplicity is $1$, which is less than the algebraic multiplicity of $2$. The matrix is **defective** (not diagonalisable).

## Visual Interpretation

- For a $2 \times 2$ real matrix, draw the transformation applied to a set of vectors arranged in a circle. The eigenvectors are the directions along which the transformed points lie along the same line from the origin (stretched or compressed). The eigenvalues give the stretch factor along these special directions.
- Complex eigenvalues mean the transformation includes a rotational component — no real direction is preserved.
- A positive eigenvalue corresponds to an eigenvector pointing in the same direction after transformation. A negative eigenvalue corresponds to an eigenvector pointing in the opposite direction (reflection).
- The condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ (for SPD matrices) measures how "stretched" the unit circle becomes — it turns into an ellipse whose major axis length is $\lambda_{\max}$ and minor axis is $\lambda_{\min}$.

## Common Mistakes

1. **Writing $|A - \lambda I| = 0$ instead of $\det(A - \lambda I) = 0$**: These mean the same thing, but the notation can confuse beginners who think it means absolute value.

2. **Forgetting that eigenvalues can be complex**: Even real matrices can have complex eigenvalues (they always occur in conjugate pairs). Students often assume eigenvalues must be real.

3. **Confusing algebraic and geometric multiplicity**: A repeated root of the characteristic polynomial does not guarantee multiple linearly independent eigenvectors.

4. **Assuming $\det(A) = \prod \lambda_i$ and $\operatorname{tr}(A) = \sum \lambda_i$ hold for non-diagonalisable matrices**: They do — these relations hold for all square matrices regardless of diagonalisability.

5. **Forgetting to subtract $\lambda$ from the diagonal only**: When forming $A - \lambda I$, the subtraction only affects the diagonal entries of $A$, not all entries.

6. **Sign errors when expanding the characteristic polynomial**: The determinant of $A - \lambda I$ expands with alternating signs; careful attention to cofactor signs is essential.

7. **Thinking eigenvalues are always non-zero**: A zero eigenvalue means the matrix is singular; this is perfectly valid and common (e.g., rank-deficient matrices).

8. **Assuming $A$ and $A^{\mathsf{T}}$ have the same eigenvectors**: They share eigenvalues (same characteristic polynomial) but generally not eigenvectors.

9. **Treating the characteristic polynomial as $|\lambda I - A| = 0$ without realising the sign**: Using $\det(\lambda I - A) = 0$ gives a monic polynomial ($\lambda^n + \dots$) instead of $(-1)^n \lambda^n + \dots$, which is often more convenient. Both are valid but lead to the same eigenvalues.

10. **Assuming that if $\lambda$ is repeated, the matrix must be defective**: Some repeated eigenvalues have full geometric multiplicity (e.g., the identity matrix has $\lambda=1$ with multiplicity $n$ and $n$ independent eigenvectors).

## Interview Questions

### Beginner

1. **Q**: What is an eigenvalue of a matrix?
   **A**: An eigenvalue $\lambda$ is a scalar such that $A\mathbf{v} = \lambda \mathbf{v}$ for some non-zero vector $\mathbf{v}$ (the eigenvector). It represents the factor by which the eigenvector is scaled when the transformation $A$ is applied.

2. **Q**: How do you find the eigenvalues of a $2 \times 2$ matrix?
   **A**: Compute $\det(A - \lambda I) = 0$, which gives a quadratic equation in $\lambda$. Solve for $\lambda$.

3. **Q**: What is the characteristic polynomial?
   **A**: $p(\lambda) = \det(A - \lambda I)$, a degree-$n$ polynomial whose roots are the eigenvalues of $A$.

4. **Q**: If $A$ is triangular, what are its eigenvalues?
   **A**: The eigenvalues of a triangular matrix are its diagonal entries.

5. **Q**: If $\lambda = 0$ is an eigenvalue of $A$, what does that tell you?
   **A**: It tells you that $A$ is singular (non-invertible), because $\det(A) = 0$ (the product of eigenvalues equals $0$).

### Intermediate

1. **Q**: Prove that if $\lambda$ is an eigenvalue of $A$, then $\lambda^2$ is an eigenvalue of $A^2$.
   **A**: If $A\mathbf{v} = \lambda \mathbf{v}$, then $A^2 \mathbf{v} = A(A\mathbf{v}) = A(\lambda \mathbf{v}) = \lambda A\mathbf{v} = \lambda^2 \mathbf{v}$. So $\mathbf{v}$ is an eigenvector of $A^2$ with eigenvalue $\lambda^2$.

2. **Q**: For a real symmetric matrix, why must all eigenvalues be real?
   **A**: Suppose $A\mathbf{v} = \lambda \mathbf{v}$ with $A^{\mathsf{T}} = A$. Take the conjugate transpose: $\mathbf{v}^* A = \bar{\lambda} \mathbf{v}^*$. Multiply: $\mathbf{v}^* A \mathbf{v} = \lambda \|\mathbf{v}\|^2$ on one hand, and $\mathbf{v}^* A \mathbf{v} = \bar{\lambda} \|\mathbf{v}\|^2$ on the other. Hence $\lambda = \bar{\lambda}$, so $\lambda$ is real.

3. **Q**: What is the relationship between eigenvalues of $A$ and the eigenvalues of $A^{-1}$?
   **A**: If $\lambda$ is an eigenvalue of $A$, then $1/\lambda$ is an eigenvalue of $A^{-1}$ (provided $\lambda \neq 0$), with the same eigenvector.

4. **Q**: When do two matrices have the same eigenvalues?
   **A**: Similar matrices ($B = P^{-1}AP$) have the same eigenvalues. In general, matrices with the same characteristic polynomial have the same eigenvalues (counting multiplicities).

5. **Q**: What is the Gershgorin circle theorem and why is it useful?
   **A**: It states that every eigenvalue of $A$ lies in at least one disc $D_i$ centred at $a_{ii}$ with radius $r_i = \sum_{j \neq i} |a_{ij}|$. It provides a quick way to bound eigenvalues without computing them, which is valuable for large matrices.

### Advanced

1. **Q**: How does the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ affect the convergence of gradient descent for a quadratic objective?
   **A**: For $f(x) = \frac12 x^{\mathsf{T}} H x$ with $H$ symmetric positive definite, gradient descent with exact line search has convergence factor $(\kappa - 1)/(\kappa + 1)$. Large $\kappa$ means slow convergence because the level sets are highly elongated ellipses, causing zigzagging. Preconditioning aims to reduce $\kappa$.

2. **Q**: In PCA, explain why the eigenvalues of the covariance matrix represent variances and why they are always non-negative.
   **A**: The covariance matrix $S = \frac1N X^{\mathsf{T}} X$ is positive semi-definite because for any $\mathbf{w}$, $\mathbf{w}^{\mathsf{T}} S \mathbf{w} = \frac1N \|X\mathbf{w}\|^2 \geq 0$. For a PSD matrix, all eigenvalues are $\geq 0$. The eigenvalue $\lambda_i$ equals $\operatorname{Var}(X \mathbf{v}_i)$, the variance of the data projected onto the $i$-th eigenvector $\mathbf{v}_i$, because $\operatorname{Var}(X \mathbf{v}_i) = \mathbf{v}_i^{\mathsf{T}} S \mathbf{v}_i = \lambda_i \|\mathbf{v}_i\|^2 = \lambda_i$.

3. **Q**: Regarding spectral normalisation in GANs, why does constraining the spectral radius of each layer's weight matrix to $1$ improve training stability?
   **A**: Spectral normalisation constrains $\rho(W) = \|W\|_2 = \sigma_{\max}(W) \leq 1$. This ensures that the Lipschitz constant of each layer is at most $1$, which prevents the discriminator from becoming too steep and producing vanishing or exploding gradients. The eigenvalues of $W^{\mathsf{T}} W$ are bounded by $1$, so the composition of layers has a controlled Lipschitz constant equal to the product of spectral radii.

## Practice Problems

### Easy

1. Find the eigenvalues of $A = \begin{bmatrix} 5 & 0 \\ 0 & 3 \end{bmatrix}$.

2. Find the eigenvalues of $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$.

3. Find the eigenvalues of $A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$.

4. If $A$ is a $3 \times 3$ matrix with eigenvalues $2$, $-1$, and $4$, what are $\det(A)$ and $\operatorname{tr}(A)$?

5. Show that $0$ is an eigenvalue of $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ without computing the characteristic polynomial.

### Medium

1. Find all eigenvalues of $A = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & -1 \end{bmatrix}$.

2. Find all eigenvalues of $A = \begin{bmatrix} 2 & 1 \\ -1 & 2 \end{bmatrix}$.

3. For $A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$ with $a, c > 0$ and $b \neq 0$, show that the eigenvalues are real.

4. Let $A$ be $3 \times 3$ with eigenvalues $1$, $1$, $3$. What are the eigenvalues of $A^2 - 2A + I$?

5. Find a $2 \times 2$ matrix whose eigenvalues are $1$ and $2$ with eigenvectors $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ respectively.

### Hard

1. Prove that if $A$ is real and skew-symmetric ($A^{\mathsf{T}} = -A$), then all eigenvalues are purely imaginary.

2. For $A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & 0 \\ 0 & 0 & 4 \end{bmatrix}$, find all eigenvalues and their algebraic and geometric multiplicities.

3. Consider the $n \times n$ matrix $A$ with $a_{ii} = 2$, $a_{i,i+1} = -1$, $a_{i+1,i} = -1$, and all other entries $0$ (the 1D discrete Laplacian). Show that the eigenvalues are $\lambda_k = 2 - 2\cos(k\pi/(n+1))$ for $k = 1, \dots, n$. What happens as $n \to \infty$?

## Solutions

### Easy Solutions

**1.** $A$ is diagonal, so eigenvalues are the diagonal entries: $\lambda_1 = 5$, $\lambda_2 = 3$.

**2.** Characteristic equation:
\[
\det\begin{bmatrix} 1-\lambda & 2 \\ 2 & 4-\lambda \end{bmatrix} = (1-\lambda)(4-\lambda) - 4 = \lambda^2 - 5\lambda + 4 - 4 = \lambda^2 - 5\lambda = \lambda(\lambda - 5) = 0.
\]
Eigenvalues: $\lambda_1 = 0$, $\lambda_2 = 5$.

**3.** $\det\begin{bmatrix} 2-\lambda & 0 \\ 0 & 2-\lambda \end{bmatrix} = (2-\lambda)^2 = 0$. Eigenvalue: $\lambda = 2$ (multiplicity 2).

**4.** $\det(A) = 2 \cdot (-1) \cdot 4 = -8$. $\operatorname{tr}(A) = 2 + (-1) + 4 = 5$.

**5.** The second column is twice the first column, so columns are linearly dependent, meaning $A$ is singular. Singular $\implies$ $\det(A) = 0$ $\implies$ product of eigenvalues $= 0$ $\implies$ at least one eigenvalue is $0$.

### Medium Solutions

**1.** $A$ is upper triangular, so eigenvalues are the diagonal entries: $\lambda_1 = 1$, $\lambda_2 = 3$, $\lambda_3 = -1$.

**2.** Characteristic equation:
\[
\det\begin{bmatrix} 2-\lambda & 1 \\ -1 & 2-\lambda \end{bmatrix} = (2-\lambda)^2 + 1 = 0.
\]
So $(2-\lambda)^2 = -1$, $2-\lambda = \pm i$, giving $\lambda = 2 \pm i$. Complex conjugate pair.

**3.** The characteristic polynomial is:
\[
\det\begin{bmatrix} a-\lambda & b \\ b & c-\lambda \end{bmatrix} = (a-\lambda)(c-\lambda) - b^2 = \lambda^2 - (a+c)\lambda + (ac - b^2).
\]
The discriminant is $\Delta = (a+c)^2 - 4(ac - b^2) = a^2 + 2ac + c^2 - 4ac + 4b^2 = (a-c)^2 + 4b^2 \geq 0$. Since $\Delta \geq 0$, both eigenvalues are real.

**4.** Since $A$ has eigenvalues $1, 1, 3$, the matrix $p(A) = A^2 - 2A + I$ has eigenvalues $p(\lambda) = \lambda^2 - 2\lambda + 1 = (\lambda - 1)^2$. So eigenvalues are: $(1-1)^2 = 0$, $(1-1)^2 = 0$, $(3-1)^2 = 4$.

**5.** Since eigenvectors are standard basis vectors, $A$ must be diagonal:
\[
A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}.
\]
Check: $A\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $A\begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix} = 2 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

### Hard Solutions

**1.** Let $A^{\mathsf{T}} = -A$ (real skew-symmetric). Take an eigenpair $A\mathbf{v} = \lambda \mathbf{v}$. Multiply both sides on the left by $\mathbf{v}^*$ (conjugate transpose): $\mathbf{v}^* A \mathbf{v} = \lambda \|\mathbf{v}\|^2$. Take the conjugate transpose of this: $\mathbf{v}^* A^{\mathsf{T}} \mathbf{v} = \bar{\lambda} \|\mathbf{v}\|^2$. Since $A^{\mathsf{T}} = -A$, we have $\mathbf{v}^*(-A) \mathbf{v} = \bar{\lambda} \|\mathbf{v}\|^2$, so $-\mathbf{v}^* A \mathbf{v} = \bar{\lambda} \|\mathbf{v}\|^2$. But $\mathbf{v}^* A \mathbf{v} = \lambda \|\mathbf{v}\|^2$, so $-\lambda \|\mathbf{v}\|^2 = \bar{\lambda} \|\mathbf{v}\|^2$. Hence $\bar{\lambda} = -\lambda$, which means $\lambda$ is purely imaginary.

**2.** Characteristic equation:
\[
\det\begin{bmatrix} 2-\lambda & 1 & 0 \\ 1 & 3-\lambda & 0 \\ 0 & 0 & 4-\lambda \end{bmatrix} = (4-\lambda)\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 3-\lambda \end{bmatrix} = 0.
\]
The $2 \times 2$ block gives: $(2-\lambda)(3-\lambda) - 1 = \lambda^2 - 5\lambda + 5 = 0$, so $\lambda = \frac{5 \pm \sqrt{5}}{2}$. So eigenvalues: $\lambda_1 = 4$, $\lambda_2 = \frac{5 + \sqrt{5}}{2} \approx 3.618$, $\lambda_3 = \frac{5 - \sqrt{5}}{2} \approx 1.382$.

All three eigenvalues are distinct, so each has algebraic multiplicity 1 and geometric multiplicity 1.

**3.** For the 1D discrete Laplacian with Dirichlet boundary conditions:
\[
A = \begin{bmatrix}
2 & -1 & 0 & \dots & 0 \\
-1 & 2 & -1 & \dots & 0 \\
0 & -1 & 2 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & 2
\end{bmatrix}.
\]
One can verify that the vector $\mathbf{v}^{(k)}$ with entries $v_j^{(k)} = \sin\left(\frac{jk\pi}{n+1}\right)$ for $j = 1, \dots, n$ satisfies:
\[
(A \mathbf{v}^{(k)})_j = 2\sin\left(\frac{jk\pi}{n+1}\right) - \sin\left(\frac{(j-1)k\pi}{n+1}\right) - \sin\left(\frac{(j+1)k\pi}{n+1}\right) = \lambda_k \sin\left(\frac{jk\pi}{n+1}\right),
\]
where $\lambda_k = 2 - 2\cos\left(\frac{k\pi}{n+1}\right) = 4\sin^2\left(\frac{k\pi}{2(n+1)}\right)$.

As $n \to \infty$, the eigenvalues fill the interval $(0, 4)$ continuously, with the smallest eigenvalue approaching $0$ (as $\lambda_1 \approx \frac{\pi^2}{(n+1)^2}$) and the largest eigenvalue approaching $4$. The condition number $\kappa = \lambda_n / \lambda_1 \approx \frac{4(n+1)^2}{\pi^2} \to \infty$, meaning the problem becomes increasingly ill-conditioned as the grid becomes finer.

## Related Concepts

- **Eigenvector (MATH-040)**: Non-zero vectors satisfying $A\mathbf{v} = \lambda \mathbf{v}$; each eigenvalue has at least one eigenvector.
- **Characteristic Polynomial**: The polynomial $\det(A - \lambda I)$ whose roots are the eigenvalues.
- **Determinant (MATH-027)**: The product of eigenvalues; a zero eigenvalue implies a zero determinant.
- **Trace (MATH-029)**: The sum of eigenvalues.
- **Diagonalization (MATH-041)**: Expressing $A = PDP^{-1}$ where $D$ contains the eigenvalues; requires $n$ linearly independent eigenvectors.
- **Spectral Theorem**: For symmetric matrices, an orthonormal eigenbasis exists with real eigenvalues.
- **Singular Value Decomposition**: Singular values are the square roots of eigenvalues of $A^{\mathsf{T}}A$ (or $AA^{\mathsf{T}}$).
- **Positive Definite Matrices**: All eigenvalues are positive.

## Next Concepts

- **Eigenvector (MATH-040)**: The vectors paired with eigenvalues; computing them via $(A - \lambda I)\mathbf{v} = \mathbf{0}$.
- **Diagonalization (MATH-041)**: Using eigenvalues and eigenvectors to factor $A$ into $PDP^{-1}$.
- **Spectral Decomposition**: Expressing a symmetric matrix as a sum of rank-1 matrices weighted by eigenvalues.
- **Singular Value Decomposition (SVD)**: Generalising eigenvalue decomposition to non-square matrices.
- **Principal Component Analysis (PCA)**: Using the eigenvalues of the covariance matrix for dimensionality reduction.
- **Jordan Normal Form**: Handling defective matrices with repeated eigenvalues.

## Summary

Eigenvalues are scalars $\lambda$ satisfying $A\mathbf{v} = \lambda \mathbf{v}$ for a non-zero vector $\mathbf{v}$. They are found by solving the characteristic equation $\det(A - \lambda I) = 0$, an $n$-degree polynomial whose roots are the eigenvalues. The set of all eigenvalues is the spectrum of $A$; the spectral radius $\rho(A) = \max|\lambda_i|$ controls stability. Key invariants are $\det(A) = \prod \lambda_i$ and $\operatorname{tr}(A) = \sum \lambda_i$. Eigenvalues determine whether a matrix is invertible (no zero eigenvalues), positive definite (all eigenvalues positive), and diagonalisable (geometric multiplicity equals algebraic multiplicity for each eigenvalue). In machine learning, eigenvalues quantify variance in PCA, control gradient descent convergence rates via the condition number, and govern stability in recurrent neural networks. The algebraic multiplicity of an eigenvalue is its multiplicity as a root of the characteristic polynomial; the geometric multiplicity is the dimension of the associated eigenspace.

## Key Takeaways

- Eigenvalues satisfy $A\mathbf{v} = \lambda \mathbf{v}$ and are the roots of $\det(A - \lambda I) = 0$.
- $\det(A) = \prod_{i=1}^n \lambda_i$ and $\operatorname{tr}(A) = \sum_{i=1}^n \lambda_i$.
- A matrix is invertible iff it has no zero eigenvalue.
- Real symmetric matrices have real eigenvalues and orthogonal eigenvectors.
- The condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ controls optimisation convergence rates.
- In PCA, eigenvalues of the covariance matrix equal the variance captured by each principal component.
- The spectral radius $\rho(A)$ determines stability of linear dynamical systems and RNN hidden state dynamics.
