# Concept: Diagonalization

## Concept ID

MATH-041

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Define diagonalization as factorising $A = PDP^{-1}$ where $D$ is diagonal.
- State and verify the condition for diagonalisability: $n$ linearly independent eigenvectors.
- Compute the diagonalisation $PDP^{-1}$ for a given matrix.
- Use diagonalisation to compute powers $A^k = PD^k P^{-1}$ efficiently.
- Interpret diagonalisation geometrically as a change-of-basis, scaling, and change-of-basis-back.
- Recognise when a matrix is not diagonalisable (defective).
- Apply diagonalisation to Markov chains, linear recurrences, and matrix exponentials.

## Prerequisites

- Eigenvalues (MATH-039) and eigenvectors (MATH-040).
- Matrix multiplication and the concept of matrix inverse.
- Basis, change of basis, and coordinate transformations (MATH-032).
- Linear independence and dimension.
- Determinants and invertibility (MATH-027).

## Definition

A square matrix $A \in \mathbb{C}^{n \times n}$ is **diagonalisable** if there exists an invertible matrix $P \in \mathbb{C}^{n \times n}$ and a diagonal matrix $D = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_n) \in \mathbb{C}^{n \times n}$ such that:

\[
A = P D P^{-1}.
\]

Equivalently, $P^{-1} A P = D$. The columns of $P$ are eigenvectors of $A$, and the diagonal entries of $D$ are the corresponding eigenvalues:

\[
P = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \dots & \mathbf{v}_n \end{bmatrix}, \quad
D = \begin{bmatrix}
\lambda_1 & 0 & \dots & 0 \\
0 & \lambda_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \lambda_n
\end{bmatrix}.
\]

The condition $A = PDP^{-1}$ is equivalent to $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$ for each column $\mathbf{v}_i$ of $P$.

## Intuition

Diagonalization expresses the idea that a linear transformation can be "untangled" into independent scaling operations along a set of special directions (the eigenvectors). The transformation $A$ is decomposed into three steps:

1. **Change to the eigenbasis** ($P^{-1}$): Express the input vector in the basis of eigenvectors.
2. **Scale each coordinate** ($D$): Multiply each eigen-coordinate by the corresponding eigenvalue.
3. **Change back to the standard basis** ($P$): Convert the result back to the original coordinate system.

This is analogous to simplifying a problem by choosing the right coordinate system. In the eigenbasis, the transformation becomes trivial — just stretching or shrinking along each axis.

If $A$ is symmetric, $P$ can be chosen to be orthogonal ($P^{-1} = P^{\mathsf{T}}$), so the change of basis is simply a rotation/reflection.

## Why This Concept Matters

Diagonalization is the single most important tool for simplifying matrix computations:

- **Matrix powers**: $A^k = PD^k P^{-1}$ reduces exponentiation to raising scalars to powers.
- **Matrix functions**: $f(A) = P f(D) P^{-1}$ where $f(D)$ applies $f$ to each diagonal entry; this enables computing matrix square roots, exponentials, and more.
- **Decoupling differential equations**: Systems of linear ODEs $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$ decouple into $n$ independent scalar equations in the eigenbasis.
- **Markov chains**: The $k$-step transition matrix $P^k$ is easily computed via diagonalisation.
- **Understanding linear layers**: The effect of repeatedly applying a neural network layer $W$ is determined by its eigenvalues.
- **Stability analysis**: The dynamics $\mathbf{x}_{k+1} = A \mathbf{x}_k$ are stable iff all $|\lambda_i| < 1$, which is immediate from the diagonal form.

## Historical Background

The concept of diagonalisation emerged from the study of quadratic forms and principal axes. **Joseph-Louis Lagrange** (1736–1813) and **Augustin-Louis Cauchy** (1789–1857) showed that a quadratic form can be reduced to a sum of squares by an appropriate change of variables — this is diagonalisation of a symmetric matrix.

**Camille Jordan** (1838–1922) developed the Jordan normal form, which generalises diagonalisation to matrices that are not diagonalisable. Jordan's work showed that every matrix can be put into an almost-diagonal form with Jordan blocks on the diagonal.

**James Joseph Sylvester** (1814–1897) stated the law of inertia: the number of positive, negative, and zero eigenvalues is invariant under congruence transformations. **Hermann Weyl** (1885–1955) extended diagonalisation to infinite-dimensional operators in quantum mechanics.

The modern matrix formulation $A = PDP^{-1}$ was standardised in mid-20th century textbooks. The QR algorithm (1961) and the divide-and-conquer algorithm for symmetric tridiagonal matrices made eigenvalue computation practical for large matrices.

## Real World Examples

### 1. Population Dynamics (Leslie Matrix)

A Leslie matrix models age-structured population growth. The population vector $\mathbf{x}_k$ evolves as $\mathbf{x}_{k+1} = L \mathbf{x}_k$. Diagonalising $L$ gives the long-term growth rate (dominant eigenvalue) and the stable age distribution (dominant eigenvector). After many generations, $\mathbf{x}_k \approx c \lambda_1^k \mathbf{v}_1$, where $\lambda_1$ is the largest eigenvalue.

### 2. PageRank and Power Iteration

Google's PageRank algorithm repeatedly applies the Google matrix $G$ to a vector: $\mathbf{v}_{k+1} = G \mathbf{v}_k$. Since $G$ is diagonalisable (it is a stochastic matrix), the iteration converges to the eigenvector corresponding to $\lambda = 1$, and the rate of convergence is governed by $|\lambda_2|^k$ where $\lambda_2$ is the second-largest eigenvalue.

### 3. Vibration Analysis — Modal Decoupling

In structural dynamics, the equations of motion $M \ddot{\mathbf{u}} + K \mathbf{u} = \mathbf{0}$ (where $M$ is mass and $K$ is stiffness) can be decoupled using the eigenvectors of $M^{-1}K$ (or solving the generalised eigenproblem). Each normal mode vibrates independently at its natural frequency — the square root of the corresponding eigenvalue.

### 4. Google's PageRank

The Google matrix is $G = \alpha S + (1-\alpha)\frac{1}{n} \mathbf{1}\mathbf{1}^{\mathsf{T}}$, where $S$ is the stochastic adjacency matrix. Diagonalisation reveals that $G$ has eigenvalues $1, \alpha \mu_2, \alpha \mu_3, \dots$, where $\mu_i$ are eigenvalues of $S$. The damping factor $\alpha = 0.85$ ensures that $|\alpha \mu_i| < 1$ for all $i \neq 1$, so power iteration converges quickly.

### 5. Economics — Leontief Input-Output Model

The Leontief input-output model involves solving $x = Ax + d$, where $A$ is the technology matrix. The solution is $x = (I - A)^{-1} d$. Diagonalising $I - A$ allows efficient computation of the Leontief inverse, especially when the model is updated repeatedly.

## AI/ML Relevance

### 1. Efficient Computation of Powers in Markov Chains

In Markov chain Monte Carlo (MCMC) and reinforcement learning, we often need the $k$-step transition matrix $P^k$. If $P$ is diagonalisable as $P = V D V^{-1}$, then $P^k = V D^k V^{-1}$. For large $k$, $D^k$ reveals the stationary distribution: the entries corresponding to $|\lambda_i| < 1$ vanish, and only the eigenvalue $1$ (and its eigenvectors) survive. The mixing time of the Markov chain is determined by the second-largest eigenvalue magnitude, also called the **spectral gap**.

Concrete example: In a random walk on a graph with adjacency matrix $A$ and degree matrix $D$, the transition matrix $P = D^{-1}A$ is often diagonalisable. The PageRank algorithm exploits this structure.

### 2. Understanding Linear Layers in Neural Networks

Consider a neural network with a repeated linear layer $W \in \mathbb{R}^{d \times d}$ followed by a non-linearity $\sigma$. The forward pass through a deep linear network (without non-linearities) is $W^k \mathbf{x}$. Diagonalising $W = PDP^{-1}$ gives $W^k \mathbf{x} = P D^k P^{-1} \mathbf{x}$. If $|\lambda_i| > 1$, the output grows exponentially with depth (exploding activations); if $|\lambda_i| < 1$, it decays (vanishing activations). This is why careful initialisation (e.g., orthogonal initialisation, setting all $|\lambda_i| = 1$) is crucial for training deep networks.

### 3. Matrix Exponentials in Continuous-Time Models

In Neural ODEs and continuous-time latent variable models, the solution to $\frac{d\mathbf{h}}{dt} = A \mathbf{h}$ is $\mathbf{h}(t) = e^{At} \mathbf{h}(0)$. If $A = PDP^{-1}$, then:

\[
e^{At} = P e^{Dt} P^{-1} = P \operatorname{diag}(e^{\lambda_1 t}, \dots, e^{\lambda_n t}) P^{-1}.
\]

This makes the computation tractable. The eigenvalues $\lambda_i$ determine whether the hidden state grows, decays, or oscillates over time. In practice, eigendecomposition of the Jacobian of an ODE flow reveals the stability properties of the learned dynamics.

### 4. PCA via Diagonalisation

The covariance matrix $S = \frac{1}{N} X^{\mathsf{T}} X$ is symmetric positive semi-definite, so it is orthogonally diagonalisable: $S = V \Lambda V^{\mathsf{T}}$. The columns of $V$ are the principal components (eigenvectors), and $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_d)$ contains the variances. The data projected onto the first $k$ components is $X V_k$.

### 5. Simplifying Quadratic Forms in SVMs

In Support Vector Machines and quadratic programming, the objective involves $\frac12 \boldsymbol{\alpha}^{\mathsf{T}} Q \boldsymbol{\alpha}$, where $Q$ is a kernel matrix. Diagonalising $Q = V \Lambda V^{\mathsf{T}}$ simplifies the optimisation and reveals the effective dimensionality of the feature space. The eigenvalues of $Q$ that are near zero correspond to redundant or noisy kernel features.

### 6. Spectral Graph Theory and Graph Neural Networks

The graph Laplacian $L = D - A$ is symmetric positive semi-definite, so $L = U \Lambda U^{\mathsf{T}}$ with $U$ orthogonal. In spectral graph convolutional networks, the convolution operator is defined as $g_\theta(L) = U g_\theta(\Lambda) U^{\mathsf{T}}$, where $g_\theta$ is a filter applied to the eigenvalues. Chebyshev polynomials approximate $g_\theta(\Lambda)$ without explicit diagonalisation.

### 7. Hessian Diagonalisation in Optimisation

Second-order optimisation methods (Newton, natural gradient) involve the Hessian matrix $H$. Diagonalising $H = V \Lambda V^{\mathsf{T}}$ reveals the curvature along different directions. The eigenvalues are the curvatures, and the eigenvectors are the principal curvature directions. Preconditioning aims to make the eigenvalue distribution more uniform (reducing the condition number).

## Mathematical Explanation

### Condition for Diagonalisability

A matrix $A \in \mathbb{C}^{n \times n}$ is diagonalisable **iff** it has $n$ linearly independent eigenvectors. This occurs exactly when for every eigenvalue, the geometric multiplicity equals the algebraic multiplicity.

Equivalently, $A$ is diagonalisable iff the sum of the dimensions of the eigenspaces equals $n$.

### The Diagonalisation Procedure

Given $A$, to compute $P$ and $D$:

1. Find all eigenvalues $\lambda_1, \dots, \lambda_n$ (roots of $\det(A - \lambda I) = 0$), counted with algebraic multiplicities.
2. For each eigenvalue $\lambda_i$, find a basis for $E_{\lambda_i} = \ker(A - \lambda_i I)$.
3. If $\sum_i \dim(E_{\lambda_i}) < n$, the matrix is not diagonalisable.
4. Assemble the eigenvectors as columns of $P$ (in any order).
5. Construct $D$ with the corresponding eigenvalues on the diagonal (in the same order).

### Verifying the Factorisation

Check that $AP = PD$. Since $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$ for each column $\mathbf{v}_i$ of $P$, we have:

\[
AP = A\begin{bmatrix} \mathbf{v}_1 & \dots & \mathbf{v}_n \end{bmatrix} =
\begin{bmatrix} \lambda_1 \mathbf{v}_1 & \dots & \lambda_n \mathbf{v}_n \end{bmatrix}
= P D.
\]

If $P$ is invertible (the eigenvectors are independent), then $A = P D P^{-1}$.

### Geometric Interpretation

The factorisation $A = P D P^{-1}$ says: "First re-coordinate into the eigenbasis ($P^{-1}$), then scale each coordinate ($D$), then re-coordinate back ($P$)." This is precisely a change-of-basis diagram:

\[
\begin{CD}
\mathbb{C}^n @>A>> \mathbb{C}^n \\
@V P^{-1} V V @AA P A\\
\mathbb{C}^n @>>D> \mathbb{C}^n
\end{CD}
\]

The map $D$ is simple (diagonal), so all complexity of $A$ is pushed into the basis change $P$.

### Diagonalisation of Symmetric Matrices

If $A$ is real symmetric, the spectral theorem guarantees that $A$ is orthogonal diagonalisable:

\[
A = Q \Lambda Q^{\mathsf{T}} = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^{\mathsf{T}},
\]

where $Q$ is orthogonal ($Q^{\mathsf{T}} = Q^{-1}$) and $\Lambda$ is real diagonal. This is called the **spectral decomposition**.

### Computing Powers

If $A = P D P^{-1}$, then:

\[
A^2 = (P D P^{-1})(P D P^{-1}) = P D^2 P^{-1},
\]
\[
A^k = P D^k P^{-1} = P \operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k) P^{-1}.
\]

This reduces the $O(n^3 \log k)$ cost of repeated multiplication to $O(n^3)$ for the diagonalisation (done once) plus $O(n^2)$ per application.

## Formula(s)

\[
A = P D P^{-1}, \quad
P = \begin{bmatrix} \mathbf{v}_1 & \dots & \mathbf{v}_n \end{bmatrix}, \quad
D = \operatorname{diag}(\lambda_1, \dots, \lambda_n)
\]

\[
P^{-1} A P = D
\]

\[
A^k = P D^k P^{-1}
\]

\[
f(A) = P f(D) P^{-1}, \quad
f(D) = \operatorname{diag}(f(\lambda_1), \dots, f(\lambda_n))
\]

\[
A = Q \Lambda Q^{\mathsf{T}} \quad \text{for symmetric } A
\]

## Properties

1. **Existence condition**: $A \in \mathbb{C}^{n \times n}$ is diagonalisable iff it has $n$ linearly independent eigenvectors.

2. **Uniqueness**: The diagonalisation is not unique. The eigenvectors can be ordered arbitrarily in $P$, and each eigenvector can be scaled by any non-zero constant.

3. **Symmetric matrices**: Every real symmetric matrix is orthogonal diagonalisable (spectral theorem).

4. **Distinct eigenvalues**: If $A$ has $n$ distinct eigenvalues, it is automatically diagonalisable.

5. **Defective matrices**: If any eigenvalue has geometric multiplicity $<$ algebraic multiplicity, $A$ is not diagonalisable.

6. **Power formula**: $A^k = P D^k P^{-1}$, where $D^k = \operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k)$.

7. **Determinant**: $\det(A) = \det(D) = \prod_{i=1}^n \lambda_i$.

8. **Trace**: $\operatorname{tr}(A) = \operatorname{tr}(D) = \sum_{i=1}^n \lambda_i$.

9. **Rank**: $\operatorname{rank}(A) = \operatorname{rank}(D)$, the number of non-zero eigenvalues.

10. **Polynomials**: $p(A) = P p(D) P^{-1}$, where $p(D) = \operatorname{diag}(p(\lambda_1), \dots, p(\lambda_n))$.

11. **Simultaneous diagonalisation**: Two diagonalisable matrices $A$ and $B$ are simultaneously diagonalisable ($A = P D_A P^{-1}$, $B = P D_B P^{-1}$) iff they commute ($AB = BA$).

12. **Orthogonal diagonalisation**: $A$ is orthogonal diagonalisable ($Q^{\mathsf{T}} A Q = \Lambda$) iff $A$ is symmetric (real case) or normal (complex case).

## Step-by-Step Worked Examples

### Example 1: Diagonalising a $2 \times 2$ Matrix

Diagonalise $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$.

**Step 1**: Find eigenvalues (from MATH-039):
\[
\lambda_1 = 2, \quad \lambda_2 = 5.
\]

**Step 2**: Find eigenvectors (from MATH-040):
For $\lambda = 2$: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$.
For $\lambda = 5$: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

**Step 3**: Form $P$ and $D$:
\[
P = \begin{bmatrix} 1 & 1 \\ -2 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix}.
\]

**Step 4**: Compute $P^{-1}$:
\[
\det(P) = 1 \cdot 1 - 1 \cdot (-2) = 3,
\]
\[
P^{-1} = \frac{1}{3} \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}.
\]

**Step 5**: Verify $A = P D P^{-1}$:
\[
P D P^{-1} = \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -2 & 1 \end{bmatrix}
\begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}
= \frac{1}{3} \begin{bmatrix} 2 & 5 \\ -4 & 5 \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}
= \frac{1}{3} \begin{bmatrix} 2+10 & -2+5 \\ -4+10 & 4+5 \end{bmatrix}
= \frac{1}{3} \begin{bmatrix} 12 & 3 \\ 6 & 9 \end{bmatrix}
= \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} = A. \quad ✓
\]

**Step 6**: Compute $A^5$:
\[
A^5 = P D^5 P^{-1}
= \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -2 & 1 \end{bmatrix}
\begin{bmatrix} 2^5 & 0 \\ 0 & 5^5 \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}
= \frac{1}{3} \begin{bmatrix} 32 & 3125 \\ -64 & 3125 \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}
= \frac{1}{3} \begin{bmatrix} 32+6250 & -32+3125 \\ -64+6250 & 64+3125 \end{bmatrix}
= \frac{1}{3} \begin{bmatrix} 6282 & 3093 \\ 6186 & 3189 \end{bmatrix}
= \begin{bmatrix} 2094 & 1031 \\ 2062 & 1063 \end{bmatrix}.
\]

---

### Example 2: Orthogonal Diagonalisation of a Symmetric $2 \times 2$

Orthogonally diagonalise $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$.

**Step 1**: Eigenvalues.
\[
\det\begin{bmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{bmatrix} = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda-2)(\lambda-4) = 0.
\]
So $\lambda_1 = 2$, $\lambda_2 = 4$.

**Step 2**: Eigenvectors.
For $\lambda = 2$:
\[
A - 2I = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \to \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}.
\]
For $\lambda = 4$:
\[
A - 4I = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}.
\]

**Step 3**: Normalise eigenvectors to get an orthogonal matrix $Q$:
\[
\mathbf{q}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \quad
\mathbf{q}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}.
\]
\[
Q = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}, \quad
\Lambda = \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix}.
\]

**Step 4**: Verify $Q^{\mathsf{T}} A Q = \Lambda$:
\[
Q^{\mathsf{T}} A Q = \frac12 \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}
= \frac12 \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 2 & 4 \\ -2 & 4 \end{bmatrix}
= \frac12 \begin{bmatrix} 4 & 0 \\ 0 & 8 \end{bmatrix}
= \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix} = \Lambda. \quad ✓
\]

**Step 5**: Spectral decomposition:
\[
A = \lambda_1 \mathbf{q}_1 \mathbf{q}_1^{\mathsf{T}} + \lambda_2 \mathbf{q}_2 \mathbf{q}_2^{\mathsf{T}}
= 2 \cdot \frac12 \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
+ 4 \cdot \frac12 \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
= \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
+ \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}
= \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}. \quad ✓
\]

---

### Example 3: Diagonalising a $3 \times 3$ Matrix

Diagonalise $A = \begin{bmatrix} 3 & -1 & 0 \\ -1 & 3 & 0 \\ 0 & 0 & 2 \end{bmatrix}$.

**Step 1**: Find eigenvalues (from MATH-039): $\lambda_1 = 2$ (multiplicity 2), $\lambda_2 = 4$ (multiplicity 1).

**Step 2**: Find eigenvectors.

For $\lambda = 2$:
\[
A - 2I = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
\to \begin{bmatrix} 1 & -1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}.
\]
Equation: $v_1 - v_2 = 0$, so $v_1 = v_2$, $v_3$ free. Basis for $E_2$:
\[
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{v}_2 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}.
\]

For $\lambda = 4$:
\[
A - 4I = \begin{bmatrix} -1 & -1 & 0 \\ -1 & -1 & 0 \\ 0 & 0 & -2 \end{bmatrix}
\to \begin{bmatrix} 1 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]
Equations: $v_1 + v_2 = 0$, $v_3 = 0$. Eigenvector: $\mathbf{v}_3 = \begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}$.

**Step 3**: Form $P$ and $D$:
\[
P = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}, \quad
D = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 4 \end{bmatrix}.
\]

**Step 4**: Compute $P^{-1}$ (using row reduction or formula):
\[
P^{-1} = \begin{bmatrix}
\frac12 & \frac12 & 0 \\
0 & 0 & 1 \\
\frac12 & -\frac12 & 0
\end{bmatrix}.
\]

**Step 5**: Verify:
\[
P D P^{-1} = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 4 \end{bmatrix}
\begin{bmatrix} \frac12 & \frac12 & 0 \\ 0 & 0 & 1 \\ \frac12 & -\frac12 & 0 \end{bmatrix}
= \begin{bmatrix} 2 & 0 & 4 \\ 2 & 0 & -4 \\ 0 & 2 & 0 \end{bmatrix}
\begin{bmatrix} \frac12 & \frac12 & 0 \\ 0 & 0 & 1 \\ \frac12 & -\frac12 & 0 \end{bmatrix}
= \begin{bmatrix} 3 & -1 & 0 \\ -1 & 3 & 0 \\ 0 & 0 & 2 \end{bmatrix} = A. \quad ✓
\]

---

### Example 4: Non-Diagonalisable (Defective) Matrix

Show that $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ is not diagonalisable.

**Step 1**: Eigenvalues: $(1-\lambda)^2 = 0$, so $\lambda = 1$ (algebraic multiplicity 2).

**Step 2**: Find eigenvectors:
\[
A - I = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}.
\]
Equation: $v_2 = 0$, $v_1$ free. Only one independent eigenvector: $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

**Step 3**: Geometric multiplicity is 1 $<$ algebraic multiplicity 2. The matrix is defective and cannot be diagonalised.

**Alternative proof**: If $A = P D P^{-1}$ existed, then $D = P^{-1} A P$ would be diagonal. But $A$ is a Jordan block; it is not similar to any diagonal matrix because any diagonal matrix similar to $A$ would have to commute with $A$, and the only matrices that commute with $A$ are polynomials in $A$, which are upper triangular with equal diagonal entries — none are diagonal (unless they are scalar multiples of $I$).

---

### Example 5: Using Diagonalisation to Solve a Recurrence

Solve $\mathbf{x}_{k+1} = A \mathbf{x}_k$ where $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$, $\mathbf{x}_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

**Step 1**: From Example 1, $A = P D P^{-1}$ with $P = \begin{bmatrix} 1 & 1 \\ -2 & 1 \end{bmatrix}$, $D = \operatorname{diag}(2, 5)$.

**Step 2**: $\mathbf{x}_k = A^k \mathbf{x}_0 = P D^k P^{-1} \mathbf{x}_0$.

**Step 3**: Compute $P^{-1} \mathbf{x}_0$:
\[
P^{-1} \mathbf{x}_0 = \frac13 \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \frac13 \begin{bmatrix} 1 \\ 2 \end{bmatrix}.
\]

**Step 4**: Multiply by $D^k$:
\[
D^k P^{-1} \mathbf{x}_0 = \frac13 \begin{bmatrix} 2^k & 0 \\ 0 & 5^k \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \frac13 \begin{bmatrix} 2^k \\ 2 \cdot 5^k \end{bmatrix}.
\]

**Step 5**: Multiply by $P$:
\[
\mathbf{x}_k = \frac13 \begin{bmatrix} 1 & 1 \\ -2 & 1 \end{bmatrix} \begin{bmatrix} 2^k \\ 2 \cdot 5^k \end{bmatrix}
= \frac13 \begin{bmatrix} 2^k + 2 \cdot 5^k \\ -2 \cdot 2^k + 2 \cdot 5^k \end{bmatrix}
= \frac13 \begin{bmatrix} 2^k + 2 \cdot 5^k \\ 2(5^k - 2^k) \end{bmatrix}.
\]

Check for $k=0$: $\mathbf{x}_0 = \frac13 \begin{bmatrix} 1 + 2 \\ 2(1-1) \end{bmatrix} = \frac13 \begin{bmatrix} 3 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$. ✓

For $k=1$: $\mathbf{x}_1 = \frac13 \begin{bmatrix} 2 + 10 \\ 2(5-2) \end{bmatrix} = \frac13 \begin{bmatrix} 12 \\ 6 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$. Check: $A \mathbf{x}_0 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$. ✓

---

### Example 6: Matrix Exponential

Compute $e^{At}$ for $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$.

**Step 1**: From Example 2, $A = Q \Lambda Q^{\mathsf{T}}$ with $Q = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$, $\Lambda = \operatorname{diag}(2, 4)$.

**Step 2**: $e^{At} = Q e^{\Lambda t} Q^{\mathsf{T}} = Q \begin{bmatrix} e^{2t} & 0 \\ 0 & e^{4t} \end{bmatrix} Q^{\mathsf{T}}$.

**Step 3**:
\[
e^{At} = \frac12 \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}
\begin{bmatrix} e^{2t} & 0 \\ 0 & e^{4t} \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
= \frac12 \begin{bmatrix} e^{2t} & e^{4t} \\ -e^{2t} & e^{4t} \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
= \frac12 \begin{bmatrix} e^{2t} + e^{4t} & -e^{2t} + e^{4t} \\ -e^{2t} + e^{4t} & e^{2t} + e^{4t} \end{bmatrix}.
\]

Thus:
\[
e^{At} = \frac12 \begin{bmatrix}
e^{2t} + e^{4t} & e^{4t} - e^{2t} \\
e^{4t} - e^{2t} & e^{2t} + e^{4t}
\end{bmatrix}.
\]

Verify at $t=0$: $e^{A \cdot 0} = \frac12 \begin{bmatrix} 1+1 & 1-1 \\ 1-1 & 1+1 \end{bmatrix} = I$. ✓

## Visual Interpretation

Geometrically, diagonalisation says that the transformation $A$ can be viewed as a composition of:

- A **rotation/reflection** ($P^{-1}$) to align with the eigenvectors.
- A **scaling** ($D$) along each eigen-direction by the corresponding eigenvalue.
- A **rotation/reflection back** ($P$) to the original coordinate system.

For a $2 \times 2$ symmetric matrix, this means that any ellipse (the image of the unit circle under $A$) has its axes aligned with the eigenvectors, and the semi-axis lengths are the eigenvalues.

For defective matrices, there are not enough independent eigenvectors. Geometrically, this corresponds to a transformation that includes shear: the eigenvectors cannot span the whole space, so the transformation has a "rotational" component in the remaining directions.

If you imagine the transformation as stretching a rubber sheet, diagonalisation identifies the directions along which the sheet is stretched (eigenvectors) and the stretch factors (eigenvalues). The eigenbasis is the coordinate system that makes the stretching purely axis-aligned.

## Common Mistakes

1. **Assuming all matrices are diagonalisable**: Many matrices are defective — they have fewer than $n$ independent eigenvectors. The shear matrix $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ is a classic counterexample.

2. **Mixing up the order of eigenvalues and eigenvectors**: The $i$-th column of $P$ must correspond to the $i$-th diagonal entry of $D$. Swapping the order in $P$ requires the same swap in $D$.

3. **Forgetting that $P$ must be invertible**: Eigenvectors must be linearly independent. If they are not, $P$ is singular and $P^{-1}$ does not exist, so $A = P D P^{-1}$ is invalid.

4. **Applying $A^k = P D^k P^{-1}$ when $A$ is not diagonalisable**: This formula requires diagonalisation. For defective matrices, use the Jordan form or other methods.

5. **Assuming $P$ is orthogonal for any diagonalisable matrix**: Only symmetric (or normal) matrices guarantee an orthogonal $P$. For general matrices, $P^{-1} \neq P^{\mathsf{T}}$.

6. **Thinking diagonalisation is unique**: The eigenvectors can be scaled arbitrarily and ordered arbitrarily. For repeated eigenvalues, any basis of the eigenspace works.

7. **Believing $A$ and $D$ are similar only if $A$ is diagonalisable**: Similarity is more general. Two matrices are similar if $B = P^{-1}AP$, regardless of whether $D$ is diagonal. The Jordan form provides a canonical non-diagonal similar form.

8. **Incorrectly computing $P^{-1}$**: Since $P$ contains eigenvectors, its inverse is often not trivial. Use row reduction or the formula for $2 \times 2$ matrices carefully.

9. **Forgetting that $A$ must be square**: Diagonalisation only applies to square matrices. Non-square matrices have the singular value decomposition instead.

10. **Assuming simultaneous diagonalisation is always possible**: Two matrices can be simultaneously diagonalised iff they commute ($AB = BA$). Non-commuting matrices generally have different eigenvectors.

## Interview Questions

### Beginner

1. **Q**: What does it mean to diagonalise a matrix?
   **A**: It means finding an invertible matrix $P$ and a diagonal matrix $D$ such that $A = P D P^{-1}$. The columns of $P$ are eigenvectors of $A$, and $D$ contains the eigenvalues.

2. **Q**: How can you tell if a matrix is diagonalisable?
   **A**: A matrix is diagonalisable iff it has $n$ linearly independent eigenvectors, i.e., the geometric multiplicity of every eigenvalue equals its algebraic multiplicity.

3. **Q**: What is $A^k$ if $A = P D P^{-1}$?
   **A**: $A^k = P D^k P^{-1}$, where $D^k$ is the diagonal matrix with entries $\lambda_i^k$.

4. **Q**: Is the identity matrix diagonalisable?
   **A**: Yes. $I = I \cdot I \cdot I^{-1}$ (any $P$ works because $I$ is already diagonal). In fact, $I$ is diagonal in any basis.

5. **Q**: What is a defective matrix?
   **A**: A matrix that cannot be diagonalised because it has at least one eigenvalue whose geometric multiplicity is less than its algebraic multiplicity. It has fewer than $n$ independent eigenvectors.

### Intermediate

1. **Q**: Prove that if $A$ has $n$ distinct eigenvalues, then $A$ is diagonalisable.
   **A**: Eigenvectors corresponding to distinct eigenvalues are linearly independent. With $n$ distinct eigenvalues, we have $n$ linearly independent eigenvectors, which form a basis. Hence $A$ is diagonalisable.

2. **Q**: Show that $A$ is diagonalisable iff $A^{\mathsf{T}}$ is diagonalisable.
   **A**: $A$ and $A^{\mathsf{T}}$ have the same characteristic polynomial, so the same eigenvalues with the same algebraic multiplicities. The geometric multiplicity of $\lambda$ for $A$ equals $\dim\ker(A - \lambda I) = n - \operatorname{rank}(A - \lambda I) = n - \operatorname{rank}((A - \lambda I)^{\mathsf{T}}) = \dim\ker(A^{\mathsf{T}} - \lambda I)$. So the geometric multiplicities match, and $A$ is diagonalisable iff $A^{\mathsf{T}}$ is.

3. **Q**: What is the relationship between diagonalisation and the spectral theorem?
   **A**: The spectral theorem says that every real symmetric matrix is orthogonally diagonalisable ($A = Q \Lambda Q^{\mathsf{T}}$ with $Q^{\mathsf{T}} = Q^{-1}$). This is a stronger statement than general diagonalisation, which only requires invertible $P$ (not necessarily orthogonal).

4. **Q**: How does diagonalisation help compute the stationary distribution of a Markov chain?
   **A**: For a Markov chain transition matrix $P$, the $k$-step distribution is $\boldsymbol{\pi}_k = \boldsymbol{\pi}_0 P^k$. If $P$ is diagonalisable ($P = V D V^{-1}$), then $\boldsymbol{\pi}_k = \boldsymbol{\pi}_0 V D^k V^{-1}$. As $k \to \infty$, $D^k$ tends to a matrix with $1$ in the position corresponding to $\lambda=1$ and $0$ elsewhere (assuming a single absorbing class), so $\boldsymbol{\pi}_\infty$ is the corresponding left eigenvector.

5. **Q**: When can two matrices be simultaneously diagonalised?
   **A**: Two diagonalisable matrices $A$ and $B$ can be simultaneously diagonalised (same $P$ for both) iff $AB = BA$. If they commute, they share a common eigenbasis.

### Advanced

1. **Q**: Explain the relationship between the Jordan normal form and diagonalisation. When would you use Jordan form instead?
   **A**: The Jordan normal form generalises diagonalisation to defective matrices. Every matrix is similar to a block diagonal matrix where each block (Jordan block) has a single eigenvalue on the diagonal and ones on the super-diagonal. Diagonalisation is the special case where all Jordan blocks are $1 \times 1$. Use Jordan form when analysing defective matrices — for example, when computing $e^{At}$ for a non-diagonalisable system, the Jordan form reveals polynomial factors in $t$ multiplied by $e^{\lambda t}$.

2. **Q**: In deep linear networks, how does the eigenvalue spectrum of the weight matrices affect the gradient dynamics during training?
   **A**: For a deep linear network $f(x) = W_L W_{L-1} \dots W_1 x$, the singular values of the product determine the network's expressivity. When training with gradient descent, the eigenvalue structure of the weight matrices influences the loss landscape. If weight matrices have eigenvalues with magnitude $>1$, the gradients can explode; if $<1$, they vanish. The Hessian of the loss has eigenvalues that are products of the weight eigenvalues, creating a highly non-convex landscape with saddle points. Saxe et al. (2014) showed that orthogonal initialisation (all eigenvalues of magnitude $1$) leads to faster, more stable training because the dynamical isometry (all singular values near $1$) preserves gradient magnitude.

3. **Q**: Prove that if $A$ and $B$ are simultaneously diagonalisable, then $AB = BA$. Is the converse true?
   **A**: If $A = P D_A P^{-1}$ and $B = P D_B P^{-1}$ with the same $P$, then $AB = P D_A D_B P^{-1} = P D_B D_A P^{-1} = BA$ because diagonal matrices commute. So simultaneous diagonalisability implies commutativity.

   The converse is true if both matrices are diagonalisable: if $AB = BA$ and both are diagonalisable, they can be simultaneously diagonalised. Proof sketch: Let $A = P D P^{-1}$. Since $AB = BA$, we have $P D P^{-1} B = B P D P^{-1}$, so $D (P^{-1} B P) = (P^{-1} B P) D$. The matrix $C = P^{-1} B P$ commutes with the diagonal matrix $D$. If $D$ has distinct eigenvalues, $C$ must be diagonal, so $B = P C P^{-1}$ is diagonalised by the same $P$. If $D$ has repeated eigenvalues, $C$ is block-diagonal, and each block can be diagonalised within its eigenspace, yielding a common eigenbasis.

   If either matrix is not diagonalisable, commutativity does not guarantee simultaneous similarity to diagonal form.

## Practice Problems

### Easy

1. Diagonalise $A = \begin{bmatrix} 5 & 0 \\ 0 & 3 \end{bmatrix}$ (this should be trivial).

2. Diagonalise $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.

3. Determine whether $A = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$ is diagonalisable.

4. If $A = P D P^{-1}$ with $P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, $D = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$, compute $A$.

5. Compute $A^{10}$ for $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$.

### Medium

1. Diagonalise $A = \begin{bmatrix} 1 & 2 \\ 3 & 2 \end{bmatrix}$ and use this to compute $A^5$.

2. Diagonalise $A = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$ over $\mathbb{C}$.

3. Let $A = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}$. Is $A$ diagonalisable? If so, find $P$ and $D$.

4. Solve the recurrence $\mathbf{x}_{k+1} = \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix} \mathbf{x}_k$ with $\mathbf{x}_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

5. For $A = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$, compute $e^{At}$ using diagonalisation.

### Hard

1. Find $A^{100}$ for $A = \begin{bmatrix} 3 & -1 \\ 2 & 0 \end{bmatrix}$ using diagonalisation.

2. A $3 \times 3$ symmetric matrix $A$ has eigenvalues $1, 2, 3$ with eigenvectors $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$, $\mathbf{v}_3 = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}$. Construct $A$.

3. Prove that $A$ is diagonalisable iff for every eigenvalue $\lambda$, $\operatorname{rank}(A - \lambda I) = \operatorname{rank}((A - \lambda I)^2)$.

## Solutions

### Easy Solutions

**1.** $A$ is already diagonal. $P = I$, $D = A$. So $A = I \cdot A \cdot I^{-1}$.

**2.** Eigenvalues:
\[
\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3) = 0.
\]
$\lambda_1 = 1$, $\lambda_2 = 3$.

For $\lambda = 1$: $A - I = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \to \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

For $\lambda = 3$: $A - 3I = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

\[
P = \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}, \quad D = \begin{bmatrix} 1 & 0 \\ 0 & 3 \end{bmatrix}, \quad
P^{-1} = \frac12 \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}.
\]

**3.** $A$ is upper triangular with eigenvalue $\lambda = 1$ (multiplicity 2). But $A - I = \begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix}$ has rank 1, so nullspace dimension is $1 < 2$. The matrix is defective — not diagonalisable.

**4.** $A = P D P^{-1}$:
\[
A = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
\begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
\cdot \frac12 \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
= \frac12 \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
= \frac12 \begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix}
= \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.
\]

**5.** $A$ is diagonal, so $A^{10} = \begin{bmatrix} 2^{10} & 0 \\ 0 & 3^{10} \end{bmatrix} = \begin{bmatrix} 1024 & 0 \\ 0 & 59049 \end{bmatrix}$.

### Medium Solutions

**1.** Eigenvalues:
\[
\det\begin{bmatrix} 1-\lambda & 2 \\ 3 & 2-\lambda \end{bmatrix} = (1-\lambda)(2-\lambda) - 6 = \lambda^2 - 3\lambda - 4 = (\lambda-4)(\lambda+1) = 0.
\]
$\lambda_1 = 4$, $\lambda_2 = -1$.

For $\lambda = 4$: $A - 4I = \begin{bmatrix} -3 & 2 \\ 3 & -2 \end{bmatrix} \to \begin{bmatrix} 1 & -\frac23 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_1 = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$.

For $\lambda = -1$: $A + I = \begin{bmatrix} 2 & 2 \\ 3 & 3 \end{bmatrix} \to \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

\[
P = \begin{bmatrix} 2 & 1 \\ 3 & -1 \end{bmatrix}, \quad
D = \begin{bmatrix} 4 & 0 \\ 0 & -1 \end{bmatrix}, \quad
P^{-1} = -\frac15 \begin{bmatrix} -1 & -1 \\ -3 & 2 \end{bmatrix} = \frac15 \begin{bmatrix} 1 & 1 \\ 3 & -2 \end{bmatrix}.
\]

\[
A^5 = P D^5 P^{-1} = \begin{bmatrix} 2 & 1 \\ 3 & -1 \end{bmatrix}
\begin{bmatrix} 4^5 & 0 \\ 0 & (-1)^5 \end{bmatrix}
\cdot \frac15 \begin{bmatrix} 1 & 1 \\ 3 & -2 \end{bmatrix}
= \frac15 \begin{bmatrix} 2 \cdot 1024 - 1 & 2 \cdot 1024 - 2 \\ 3 \cdot 1024 + 1 & 3 \cdot 1024 - 2 \end{bmatrix}
= \frac15 \begin{bmatrix} 2047 & 2046 \\ 3073 & 3070 \end{bmatrix}.
\]

**2.** Eigenvalues:
\[
\det\begin{bmatrix} 1-\lambda & -1 \\ 1 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 + 1 = \lambda^2 - 2\lambda + 2 = 0.
\]
$\lambda = 1 \pm i$.

For $\lambda = 1+i$:
\[
A - (1+i)I = \begin{bmatrix} -i & -1 \\ 1 & -i \end{bmatrix} \to \begin{bmatrix} 1 & -i \\ 0 & 0 \end{bmatrix}.
\]
Eigenvector: $\mathbf{v}_1 = \begin{bmatrix} i \\ 1 \end{bmatrix}$.

For $\lambda = 1-i$: $\mathbf{v}_2 = \begin{bmatrix} -i \\ 1 \end{bmatrix}$.

\[
P = \begin{bmatrix} i & -i \\ 1 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 1+i & 0 \\ 0 & 1-i \end{bmatrix}.
\]

**3.** Eigenvalues: upper triangular, so $\lambda = 2$ (multiplicity 2), $\lambda = 3$ (multiplicity 1).

For $\lambda = 2$:
\[
A - 2I = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}.
\]
Rows: $v_2 = 0$, $v_3 = 0$, $v_1$ free. Only one eigenvector: $\mathbf{v} = \mathbf{e}_1$. Geometric multiplicity $1 < 2$, so $A$ is **not** diagonalisable.

**4.** Eigenvalues:
\[
\det\begin{bmatrix} 3-\lambda & 2 \\ 2 & 3-\lambda \end{bmatrix} = (3-\lambda)^2 - 4 = \lambda^2 - 6\lambda + 5 = (\lambda-1)(\lambda-5) = 0.
\]
$\lambda_1 = 1$, $\lambda_2 = 5$.

$\mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

\[
P = \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 1 & 0 \\ 0 & 5 \end{bmatrix}, \quad
P^{-1} = \frac12 \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}.
\]

\[
\mathbf{x}_k = P D^k P^{-1} \mathbf{x}_0
= \frac12 \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}
\begin{bmatrix} 1^k & 0 \\ 0 & 5^k \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \end{bmatrix}
= \frac12 \begin{bmatrix} 1 & 5^k \\ -1 & 5^k \end{bmatrix}
\begin{bmatrix} 1 \\ 1 \end{bmatrix}
= \frac12 \begin{bmatrix} 1 + 5^k \\ -1 + 5^k \end{bmatrix}.
\]

**5.** Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = 2$. Eigenvectors:
For $\lambda=1$: $A - I = \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix} \to \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.
For $\lambda=2$: $A - 2I = \begin{bmatrix} -1 & 1 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

\[
P = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}, \quad
P^{-1} = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}.
\]

\[
e^{At} = P e^{Dt} P^{-1}
= \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} e^t & 0 \\ 0 & e^{2t} \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}
= \begin{bmatrix} e^t & e^{2t} \\ 0 & e^{2t} \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}
= \begin{bmatrix} e^t & e^{2t} - e^t \\ 0 & e^{2t} \end{bmatrix}.
\]

Check at $t=0$: $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I$. ✓

### Hard Solutions

**1.** Eigenvalues:
\[
\det\begin{bmatrix} 3-\lambda & -1 \\ 2 & -\lambda \end{bmatrix} = \lambda(\lambda-3) + 2 = \lambda^2 - 3\lambda + 2 = (\lambda-1)(\lambda-2) = 0.
\]
$\lambda_1 = 1$, $\lambda_2 = 2$.

For $\lambda = 1$: $A - I = \begin{bmatrix} 2 & -1 \\ 2 & -1 \end{bmatrix} \to \begin{bmatrix} 1 & -\frac12 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$.

For $\lambda = 2$: $A - 2I = \begin{bmatrix} 1 & -1 \\ 2 & -2 \end{bmatrix} \to \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

\[
P = \begin{bmatrix} 1 & 1 \\ 2 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}, \quad
P^{-1} = \begin{bmatrix} -1 & 1 \\ 2 & -1 \end{bmatrix}.
\]

\[
A^{100} = P D^{100} P^{-1}
= \begin{bmatrix} 1 & 1 \\ 2 & 1 \end{bmatrix}
\begin{bmatrix} 1^{100} & 0 \\ 0 & 2^{100} \end{bmatrix}
\begin{bmatrix} -1 & 1 \\ 2 & -1 \end{bmatrix}
= \begin{bmatrix} 1 & 2^{100} \\ 2 & 2^{100} \end{bmatrix}
\begin{bmatrix} -1 & 1 \\ 2 & -1 \end{bmatrix}
= \begin{bmatrix} -1 + 2 \cdot 2^{100} & 1 - 2^{100} \\ -2 + 2 \cdot 2^{100} & 2 - 2^{100} \end{bmatrix}.
\]

**2.** Normalise the eigenvectors (they are already orthogonal: $\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$, $\mathbf{v}_2 \cdot \mathbf{v}_3 = 0$, $\mathbf{v}_1 \cdot \mathbf{v}_3 = 1 \cdot 1 + 0 \cdot 0 + 1 \cdot (-1) = 0$):

\[
\mathbf{q}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad
\mathbf{q}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{q}_3 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}.
\]

\[
Q = \begin{bmatrix}
\frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
0 & 1 & 0 \\
\frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
\end{bmatrix}, \quad
\Lambda = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}.
\]

$A = Q \Lambda Q^{\mathsf{T}}$ gives:
\[
A = \begin{bmatrix}
\frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
0 & 1 & 0 \\
\frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
\end{bmatrix}
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}
\begin{bmatrix}
\frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
0 & 1 & 0 \\
\frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
\end{bmatrix}.
\]
\[
A = \begin{bmatrix}
\frac12 + \frac32 & 0 & \frac12 - \frac32 \\
0 & 2 & 0 \\
\frac12 - \frac32 & 0 & \frac12 + \frac32
\end{bmatrix}
= \begin{bmatrix}
2 & 0 & -1 \\
0 & 2 & 0 \\
-1 & 0 & 2
\end{bmatrix}.
\]

**3.** ($\Rightarrow$) If $A$ is diagonalisable, then $A - \lambda I$ is similar to $D - \lambda I$, which has eigenvalues $\lambda_i - \lambda$. The rank of $A - \lambda I$ equals the number of non-zero eigenvalues $\lambda_i - \lambda$, which is $n - \dim(E_\lambda)$. Since $A - \lambda I$ is diagonalisable (same eigenvectors as $A$), $\operatorname{rank}((A - \lambda I)^2) = \sum_{i: \lambda_i \neq \lambda} (\lambda_i - \lambda)^2 = \operatorname{rank}(A - \lambda I)$.

($\Leftarrow$) If $\operatorname{rank}(A - \lambda I) = \operatorname{rank}((A - \lambda I)^2)$, then the Jordan blocks for $\lambda$ are all $1 \times 1$ (no nilpotent part). This holds for every eigenvalue, so $A$ is diagonalisable.

## Related Concepts

- **Eigenvalue (MATH-039)**: The scalars $\lambda_i$ that appear on the diagonal of $D$.
- **Eigenvector (MATH-040)**: The columns of $P$ are eigenvectors of $A$.
- **Change of Basis (MATH-032)**: $P$ implements a change of basis from the standard basis to the eigenbasis.
- **Spectral Theorem**: The special case of orthogonal diagonalisation for symmetric matrices.
- **Jordan Normal Form**: Generalisation of diagonalisation to defective matrices, using Jordan blocks.
- **Singular Value Decomposition (SVD)**: Generalisation to non-square matrices; $A = U \Sigma V^{\mathsf{T}}$.
- **Matrix Exponential**: $e^{At} = P e^{Dt} P^{-1}$, used in solving linear ODEs.
- **Cayley-Hamilton Theorem**: Every matrix satisfies its own characteristic polynomial, providing an alternative way to compute powers.
- **Similarity Transform**: $B = P^{-1} A P$; diagonalisation is finding $P$ that makes $B$ diagonal.

## Next Concepts

- **Jordan Normal Form**: Handling defective matrices with generalised eigenvectors.
- **Singular Value Decomposition (SVD)**: Diagonalisation for non-square matrices.
- **Spectral Theorem**: Orthogonal diagonalisation of symmetric and normal matrices.
- **Simultaneous Diagonalisation**: When two matrices share an eigenbasis.
- **Positive Definite Matrices**: All eigenvalues positive; $A = Q \Lambda Q^{\mathsf{T}}$ with $\lambda_i > 0$.
- **Principal Component Analysis**: Orthogonal diagonalisation of the covariance matrix.
- **Matrix Functions**: $f(A) = P f(D) P^{-1}$ for analytic functions $f$, including the matrix sign function, square root, and exponential.

## Summary

Diagonalisation represents a square matrix $A$ as $A = P D P^{-1}$, where $D$ is diagonal and $P$ contains the eigenvectors of $A$ as columns. This factorisation exists iff $A$ has $n$ linearly independent eigenvectors — i.e., the geometric multiplicity of every eigenvalue equals its algebraic multiplicity. Diagonalisation simplifies all matrix computations that use powers, including $A^k = P D^k P^{-1}$, $e^{At} = P e^{Dt} P^{-1}$, and more generally $f(A) = P f(D) P^{-1}$. For symmetric matrices, $P$ can be chosen orthogonal, yielding the spectral decomposition $A = Q \Lambda Q^{\mathsf{T}}$. Matrices that are not diagonalisable are called defective and require the Jordan normal form. Diagonalisation is essential for decoupling linear dynamical systems, analysing Markov chain convergence, computing matrix exponentials for continuous-time models, and understanding the behaviour of repeated linear transformations in deep neural networks.

## Key Takeaways

- $A = P D P^{-1}$ where $D$ is diagonal containing eigenvalues, $P$ has eigenvectors as columns.
- $A$ is diagonalisable iff it has $n$ linearly independent eigenvectors.
- $A^k = P D^k P^{-1}$ dramatically simplifies computing powers.
- Symmetric matrices are orthogonally diagonalisable: $A = Q \Lambda Q^{\mathsf{T}}$.
- Defective matrices lack a full eigenbasis and cannot be diagonalised.
- Diagonalisation decouples linear recurrences and differential equations.
- In ML, diagonalisation enables efficient Markov chain computations, analysis of linear layer dynamics, and spectral decomposition of covariance and kernel matrices.
- The spectral gap (difference between the largest and second-largest eigenvalue magnitude) controls convergence rates of Markov chains and iterative algorithms.
