# Concept: Trace

## Concept ID

MATH-029

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define the trace of a square matrix as the sum of its diagonal entries.
- Compute the trace of matrices of various sizes.
- State and apply the key algebraic properties of the trace operator.
- Relate the trace to the sum of eigenvalues.
- Use the trace to compute the Frobenius norm of a matrix.
- Understand how the trace appears in PCA, metric learning, and loss functions in machine learning.

## Prerequisites

- Basic matrix notation and operations (addition, multiplication, transposition).
- Familiarity with eigenvalues and eigenvectors at an introductory level.
- Familiarity with vector norms (optional but helpful).

## Definition

Let $A$ be an $n \times n$ square matrix with entries $a_{ij}$. The **trace** of $A$, denoted $\text{tr}(A)$, is the sum of its diagonal entries:

$$
\text{tr}(A) = \sum_{i=1}^{n} a_{ii}
$$

For example, if

$$
A = \begin{pmatrix} 2 & 7 & 1 \\ 0 & 5 & -3 \\ 4 & 1 & 8 \end{pmatrix},
\qquad
\text{tr}(A) = 2 + 5 + 8 = 15.
$$

The trace is only defined for square matrices because only square matrices have a principal diagonal from the top-left to the bottom-right.

## Intuition

Think of the trace as a single number that "summarises" the diagonal of a matrix. While the diagonal alone does not capture the whole matrix, the trace is surprisingly useful because it is invariant under many transformations (like cyclic permutations of matrix products) and connects directly to eigenvalues.

A helpful mental model: if a matrix represents a linear transformation, the trace is (up to sign) the sum of the "stretching factors" along the eigen-directions — that is, the sum of the eigenvalues. This makes it a compact scalar fingerprint of the transformation.

## Why This Concept Matters

The trace arises in nearly every branch of mathematics that uses matrices:

- **Linear Algebra:** It connects to eigenvalues, determinants, and matrix invariants.
- **Calculus:** The derivative of a scalar function of a matrix often involves the trace (e.g., $\frac{d}{dX} \text{tr}(AX) = A^T$).
- **Statistics:** The trace of a covariance matrix equals the total variance in a dataset.
- **Machine Learning:** Loss functions, regularisation terms, and norm computations frequently use the trace because it is linear and easy to differentiate.

Without the trace, expressing the Frobenius norm, computing total variance, or deriving gradients in matrix calculus would be far more cumbersome.

## Historical Background

The trace (from Latin *tractus*, meaning "drawn out") has been implicitly used since the early days of matrix theory in the 19th century. The German mathematician Ferdinand Georg Frobenius (1849–1917) was among the first to systematically study matrix invariants, including the trace. The term "trace" itself (German: *Spur*) was popularised by David Hilbert and his school in the early 1900s in the context of integral equations and infinite-dimensional spaces. The trace class operators in functional analysis are a direct generalisation.

## Real World Examples

1. **Structural engineering:** The sum of the diagonal entries of a stress tensor (the trace) is proportional to the hydrostatic pressure at a point in a material.
2. **Quantum mechanics:** The trace of the density matrix is always 1 (normalisation of probabilities). The expectation value of an observable $\hat{O}$ is $\langle \hat{O} \rangle = \text{tr}(\rho \hat{O})$.
3. **Network analysis:** The trace of the adjacency matrix of a graph counts the number of self-loops. The trace of powers of the adjacency matrix counts closed walks of a given length.
4. **Economics:** In input-output models, the trace of the Leontief matrix relates to the total output multipliers.

## AI/ML Relevance

The trace is a workhorse in machine learning:

1. **Frobenius norm:** The most common matrix norm in ML is the Frobenius norm:
   $$
   \|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^T A)}
   $$
   This is used in regularisation (weight decay), matrix completion (e.g., recommender systems), and measuring reconstruction error.

2. **Total variance in PCA:** Given a data matrix $X \in \mathbb{R}^{n \times d}$ (centred), the sample covariance matrix is $S = \frac{1}{n-1} X^T X$. The total variance is
   $$
   \text{tr}(S) = \sum_{i=1}^{d} \lambda_i
   $$
   where $\lambda_i$ are the eigenvalues of $S$. This tells us how much total spread exists in the data. In PCA, the fraction of variance explained by the first $k$ principal components is $(\lambda_1 + \dots + \lambda_k) / \text{tr}(S)$.

3. **Metric learning:** Some loss functions use the trace to encourage certain matrix properties. For example, in learning a Mahalanobis distance $d_M(x, y) = (x-y)^T M (x-y)$ with $M$ positive semidefinite, a regularisation term $\text{tr}(M)$ may be added to control complexity.

4. **Deep learning:** The trace of the Fisher information matrix appears in natural gradient descent. The trace of the Hessian is used in some optimisation and pruning algorithms.

5. **Reinforcement learning:** The trace of a matrix plays a role in the eligibility trace algorithm (though the name "trace" there refers to a different concept — a decaying record of visited states).

## Mathematical Explanation

The trace is a linear functional on the vector space of $n \times n$ matrices. It satisfies:

- **Linearity:** $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$ and $\text{tr}(cA) = c \,\text{tr}(A)$ for scalar $c$.
- **Invariance under transposition:** $\text{tr}(A^T) = \text{tr}(A)$.
- **Cyclic property:** $\text{tr}(AB) = \text{tr}(BA)$ for any matrices $A$ and $B$ where the products are defined. More generally, $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$ — the trace is invariant under cyclic permutations.
- **Similarity invariance:** If $P$ is invertible, then $\text{tr}(P^{-1}AP) = \text{tr}(A)$. This means the trace is a *similarity invariant*.
- **Sum of eigenvalues:** If $\lambda_1, \dots, \lambda_n$ are the eigenvalues of $A$ (counted with multiplicity), then $\text{tr}(A) = \sum_{i=1}^n \lambda_i$.
- **Relation to determinant:** For a $2 \times 2$ matrix, the characteristic polynomial is $\lambda^2 - \text{tr}(A)\lambda + \det(A)$. For larger matrices, the trace appears as the coefficient of $\lambda^{n-1}$ (with a sign).

## Formula(s)

| Formula | Description |
|---------|-------------|
| $\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$ | Definition |
| $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$ | Additivity |
| $\text{tr}(cA) = c \,\text{tr}(A)$ | Homogeneity |
| $\text{tr}(A^T) = \text{tr}(A)$ | Transpose invariance |
| $\text{tr}(AB) = \text{tr}(BA)$ | Cyclic property (2 matrices) |
| $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$ | Cyclic property (3 matrices) |
| $\text{tr}(P^{-1}AP) = \text{tr}(A)$ | Similarity invariance |
| $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$ | Sum of eigenvalues |
| $\|A\|_F = \sqrt{\text{tr}(A^T A)}$ | Frobenius norm |

## Properties

- **Linearity:** The trace is a linear map from $\mathbb{R}^{n \times n}$ to $\mathbb{R}$.
- **Cyclic invariance:** $\text{tr}(AB) = \text{tr}(BA)$ even though $AB \neq BA$ in general. Note that this does NOT extend to arbitrary permutations: $\text{tr}(ABC) \neq \text{tr}(ACB)$ in general.
- **Does not depend on basis:** $\text{tr}(P^{-1}AP) = \text{tr}(A)$ means you can change coordinates freely without changing the trace.
- **Eigenvalue connection:** This is one of the most useful properties: you can compute the sum of eigenvalues without ever finding the eigenvalues individually.
- **Trace of a product is an inner product:** On the space of matrices, $\langle A, B \rangle = \text{tr}(A^T B)$ defines the Frobenius inner product, making the space of matrices into an inner product space.

## Step-by-Step Worked Examples

### Example 1: Basic computation and linearity

Let
$$
A = \begin{pmatrix} 3 & -1 & 0 \\ 2 & 4 & 5 \\ -2 & 1 & 6 \end{pmatrix},
\qquad
B = \begin{pmatrix} 1 & 2 & -3 \\ 0 & -1 & 4 \\ 2 & 3 & 0 \end{pmatrix}.
$$

Compute $\text{tr}(A)$, $\text{tr}(B)$, $\text{tr}(A+B)$, and verify linearity.

**Solution:**

$\text{tr}(A) = 3 + 4 + 6 = 13$

$\text{tr}(B) = 1 + (-1) + 0 = 0$

$A + B = \begin{pmatrix} 3+1 & -1+2 & 0+(-3) \\ 2+0 & 4+(-1) & 5+4 \\ -2+2 & 1+3 & 6+0 \end{pmatrix} = \begin{pmatrix} 4 & 1 & -3 \\ 2 & 3 & 9 \\ 0 & 4 & 6 \end{pmatrix}$

$\text{tr}(A+B) = 4 + 3 + 6 = 13$

Check: $\text{tr}(A) + \text{tr}(B) = 13 + 0 = 13 \quad\checkmark$

### Example 2: Cyclic property $\text{tr}(AB) = \text{tr}(BA)$

Let
$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix},
\qquad
B = \begin{pmatrix} 0 & 1 \\ -1 & 1 \end{pmatrix}.
$$

Compute $AB$, $BA$, and verify $\text{tr}(AB) = \text{tr}(BA)$.

**Solution:**

$$
AB = \begin{pmatrix} 1(0) + 2(-1) & 1(1) + 2(1) \\ 3(0) + 4(-1) & 3(1) + 4(1) \end{pmatrix}
= \begin{pmatrix} -2 & 3 \\ -4 & 7 \end{pmatrix}
$$

$\text{tr}(AB) = -2 + 7 = 5$

$$
BA = \begin{pmatrix} 0(1) + 1(3) & 0(2) + 1(4) \\ -1(1) + 1(3) & -1(2) + 1(4) \end{pmatrix}
= \begin{pmatrix} 3 & 4 \\ 2 & 2 \end{pmatrix}
$$

$\text{tr}(BA) = 3 + 2 = 5 \quad\checkmark$

Note that $AB \neq BA$ but their traces are equal.

### Example 3: Trace equals sum of eigenvalues

Let
$$
A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}.
$$

Find the eigenvalues and verify that $\text{tr}(A) = \lambda_1 + \lambda_2$.

**Solution:**

Characteristic polynomial:
$$
\det(A - \lambda I) = \det\begin{pmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{pmatrix}
= (4-\lambda)(3-\lambda) - 2
= \lambda^2 - 7\lambda + 10
$$

Eigenvalues:
$$
\lambda = \frac{7 \pm \sqrt{49 - 40}}{2} = \frac{7 \pm 3}{2}
$$

So $\lambda_1 = 5$, $\lambda_2 = 2$.

$\text{tr}(A) = 4 + 3 = 7$

$\lambda_1 + \lambda_2 = 5 + 2 = 7 \quad\checkmark$

### Example 4: Frobenius norm via trace

Let
$$
A = \begin{pmatrix} 1 & -2 & 0 \\ 3 & 1 & 4 \end{pmatrix}.
$$

Compute $\|A\|_F$ directly and via $\text{tr}(A^T A)$.

**Solution:**

Direct: $\|A\|_F = \sqrt{1^2 + (-2)^2 + 0^2 + 3^2 + 1^2 + 4^2} = \sqrt{1 + 4 + 0 + 9 + 1 + 16} = \sqrt{31}$

Now via trace:

$$
A^T A = \begin{pmatrix} 1 & 3 \\ -2 & 1 \\ 0 & 4 \end{pmatrix}
\begin{pmatrix} 1 & -2 & 0 \\ 3 & 1 & 4 \end{pmatrix}
= \begin{pmatrix}
1(1)+3(3) & 1(-2)+3(1) & 1(0)+3(4) \\
-2(1)+1(3) & -2(-2)+1(1) & -2(0)+1(4) \\
0(1)+4(3) & 0(-2)+4(1) & 0(0)+4(4)
\end{pmatrix}
$$

$$
A^T A = \begin{pmatrix}
10 & 1 & 12 \\
1 & 5 & 4 \\
12 & 4 & 16
\end{pmatrix}
$$

$\text{tr}(A^T A) = 10 + 5 + 16 = 31$

So $\|A\|_F = \sqrt{31} \quad\checkmark$

### Example 5: Trace in PCA — total variance

A dataset of 3 samples with 2 features (already centred):

$$
X = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ -1 & -1 \end{pmatrix}
$$

Compute the covariance matrix $S = \frac{1}{n-1} X^T X$ and its trace (total variance).

**Solution:**

$n = 3$, so $n-1 = 2$.

$$
X^T X = \begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \end{pmatrix}
\begin{pmatrix} 2 & -1 \\ -1 & 2 \\ -1 & -1 \end{pmatrix}
= \begin{pmatrix}
2(2)+(-1)(-1)+(-1)(-1) & 2(-1)+(-1)(2)+(-1)(-1) \\
-1(2)+2(-1)+(-1)(-1) & -1(-1)+2(2)+(-1)(-1)
\end{pmatrix}
$$

$$
X^T X = \begin{pmatrix}
4+1+1 & -2-2+1 \\
-2-2+1 & 1+4+1
\end{pmatrix}
= \begin{pmatrix}
6 & -3 \\
-3 & 6
\end{pmatrix}
$$

$$
S = \frac{1}{2} \begin{pmatrix} 6 & -3 \\ -3 & 6 \end{pmatrix} = \begin{pmatrix} 3 & -1.5 \\ -1.5 & 3 \end{pmatrix}
$$

Total variance = $\text{tr}(S) = 3 + 3 = 6$.

Eigenvalues of $S$: $\det\begin{pmatrix} 3-\lambda & -1.5 \\ -1.5 & 3-\lambda \end{pmatrix} = (3-\lambda)^2 - 2.25 = \lambda^2 - 6\lambda + 9 - 2.25 = \lambda^2 - 6\lambda + 6.75$

$\lambda = \frac{6 \pm \sqrt{36 - 27}}{2} = \frac{6 \pm 3}{2}$, so $\lambda_1 = 4.5$, $\lambda_2 = 1.5$.

Sum of eigenvalues $= 4.5 + 1.5 = 6 = \text{tr}(S) \quad\checkmark$

The first PC explains $\frac{4.5}{6} = 75\%$ of total variance.

## Visual Interpretation

Imagine a square matrix as a grid of numbers. The trace picks out the diagonal cells and adds them up.

For a $2 \times 2$ matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the trace $a+d$ controls the "average scaling" of the transformation.

Consider how the trace behaves under rotations (orthogonal transformations): because $\text{tr}(Q^T A Q) = \text{tr}(A)$ for any orthogonal $Q$, the trace is invariant when you spin the coordinate axes. This makes it a coordinate-independent scalar descriptor of a linear map.

If you visualise a matrix as a linear transformation stretching and rotating space, the trace (sum of eigenvalues) captures the net expansion or contraction (ignoring rotation). For example, a pure rotation matrix in 2D has trace $2\cos\theta$, which is zero only when $\theta = 90^\circ$ — the trace gives information about the rotation angle.

## Common Mistakes

1. **Taking trace of a non-square matrix.** The trace is only defined for square matrices. Given a $3 \times 2$ matrix, you cannot compute its trace.

2. **Assuming $\text{tr}(AB) = \text{tr}(A)\,\text{tr}(B)$.** This is false. For example, $A = I$, $B = I$: $\text{tr}(AB) = n$, but $\text{tr}(A)\,\text{tr}(B) = n^2$.

3. **Thinking $\text{tr}(ABC) = \text{tr}(ACB)$.** Cyclic permutations preserve the trace, but arbitrary permutations do not. For example, let $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, B = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, C = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$. Then $\text{tr}(ABC) = 0$ but $\text{tr}(ACB) = 1$.

4. **Confusing trace with determinant.** The trace is the sum of diagonal entries; the determinant is a more complicated polynomial. Both are similarity invariants, but they encode different information.

5. **Forgetting the factor in the Frobenius norm.** $\|A\|_F = \sqrt{\text{tr}(A^T A)}$, not $\text{tr}(A^T A)$ itself.

## Interview Questions

### Beginner

**Q1:** What is the trace of a $3 \times 3$ identity matrix?
**A1:** $\text{tr}(I_3) = 1 + 1 + 1 = 3$.

**Q2:** If $A = \begin{pmatrix} 5 & 2 \\ -1 & 3 \end{pmatrix}$, what is $\text{tr}(A)$?
**A2:** $\text{tr}(A) = 5 + 3 = 8$.

**Q3:** Is the trace defined for a $2 \times 3$ matrix?
**A3:** No. The trace is only defined for square matrices.

**Q4:** True or false: $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$.
**A4:** True. This is the additivity property of the trace.

**Q5:** What is $\text{tr}(A^T)$ in terms of $\text{tr}(A)$?
**A5:** $\text{tr}(A^T) = \text{tr}(A)$ because transposition does not change the diagonal entries.

### Intermediate

**Q1:** Show that $\text{tr}(AB) = \text{tr}(BA)$ for $2 \times 2$ matrices.
**A1:** Let $A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$, $B = \begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{pmatrix}$. Then $\text{tr}(AB) = a_{11}b_{11} + a_{12}b_{21} + a_{21}b_{12} + a_{22}b_{22}$. $\text{tr}(BA) = b_{11}a_{11} + b_{12}a_{21} + b_{21}a_{12} + b_{22}a_{22}$. These are the same sum (terms reordered).

**Q2:** If $\text{tr}(A) = 0$, does it imply $A$ is singular (determinant zero)?
**A2:** No. For example, $A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ has $\text{tr}(A) = 0$ but $\det(A) = -1 \neq 0$.

**Q3:** How is the trace related to the Frobenius norm?
**A3:** $\|A\|_F = \sqrt{\text{tr}(A^T A)}$.

**Q4:** What does the trace of a covariance matrix represent in statistics?
**A4:** It represents the total variance across all dimensions.

**Q5:** Prove that the trace is invariant under similarity transformations.
**A5:** $\text{tr}(P^{-1}AP) = \text{tr}((P^{-1}A)P) = \text{tr}(P(P^{-1}A)) = \text{tr}((PP^{-1})A) = \text{tr}(A)$. (Using the cyclic property.)

### Advanced

**Q1:** Derive the derivative of $\text{tr}(AX)$ with respect to $X$.
**A1:** $\text{tr}(AX) = \sum_i \sum_j a_{ij} x_{ji}$. Then $\frac{\partial}{\partial x_{kl}} \text{tr}(AX) = a_{lk}$. So $\frac{d}{dX} \text{tr}(AX) = A^T$.

**Q2:** Show that there is no matrix $A$ such that $AB - BA = I$ for finite-dimensional matrices (this is related to the uncertainty principle in quantum mechanics).
**A2:** Take the trace of both sides: $\text{tr}(AB - BA) = \text{tr}(AB) - \text{tr}(BA) = 0$, but $\text{tr}(I) = n \neq 0$. Contradiction. So no such finite-dimensional matrices exist.

**Q3:** How would you compute $\text{tr}(e^A)$ without computing the full matrix exponential?
**A3:** $\text{tr}(e^A) = \sum_i e^{\lambda_i}$, where $\lambda_i$ are the eigenvalues of $A$. This follows because $e^A$ has eigenvalues $e^{\lambda_i}$, and the trace equals the sum of eigenvalues.

## Practice Problems

### Easy

**E1:** Compute $\text{tr}(A)$ for $A = \begin{pmatrix} 7 & 0 & 0 \\ 0 & -3 & 0 \\ 0 & 0 & 2 \end{pmatrix}$.

**E2:** If $\text{tr}(A) = 5$ and $\text{tr}(B) = -2$, what is $\text{tr}(A + B)$?

**E3:** What is $\text{tr}(5I_4)$?

**E4:** Compute $\text{tr}(A)$ where $A = \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}$.

**E5:** For $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, compute $\text{tr}(A^T)$.

### Medium

**M1:** Let $A = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}$ and $B = \begin{pmatrix} 1 & 0 \\ 2 & -2 \end{pmatrix}$. Compute $\text{tr}(AB)$ and $\text{tr}(BA)$ and verify they are equal.

**M2:** For $A = \begin{pmatrix} 1 & 2 \\ 0 & -1 \end{pmatrix}$, find the eigenvalues and verify $\text{tr}(A) = \lambda_1 + \lambda_2$.

**M3:** Compute the Frobenius norm of $A = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}$ using the trace formula.

**M4:** Let $A$ be $2 \times 2$ with $\text{tr}(A) = 4$ and $\det(A) = 3$. Find the eigenvalues.

**M5:** If $A$ is an $n \times n$ matrix and $\text{tr}(A) = 0$, what can you say about the sum of its eigenvalues?

### Hard

**H1:** Prove that $\text{tr}(A^T A) = 0$ if and only if $A = 0$.

**H2:** Suppose $A$ is a real symmetric $n \times n$ matrix. Show that $\text{tr}(A^2) \geq \frac{(\text{tr}(A))^2}{n}$. When does equality hold?

**H3:** Let $A$ and $B$ be $n \times n$ matrices. Prove that $\text{tr}((A+B)^T (A+B)) = \text{tr}(A^T A) + 2\text{tr}(A^T B) + \text{tr}(B^T B)$.

## Solutions

### Easy Solutions

**E1:** $\text{tr}(A) = 7 + (-3) + 2 = 6$.

**E2:** $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B) = 5 + (-2) = 3$.

**E3:** $5I_4$ is a $4 \times 4$ diagonal matrix with all diagonal entries 5. So $\text{tr}(5I_4) = 5 \times 4 = 20$.

**E4:** $\text{tr}(A) = a + e + i$.

**E5:** $\text{tr}(A^T) = \text{tr}(A) = 1 + 4 = 5$.

### Medium Solutions

**M1:**
$$
AB = \begin{pmatrix} 2(1)+1(2) & 2(0)+1(-2) \\ -1(1)+3(2) & -1(0)+3(-2) \end{pmatrix}
= \begin{pmatrix} 4 & -2 \\ 5 & -6 \end{pmatrix}
$$
$\text{tr}(AB) = 4 + (-6) = -2$.

$$
BA = \begin{pmatrix} 1(2)+0(-1) & 1(1)+0(3) \\ 2(2)+(-2)(-1) & 2(1)+(-2)(3) \end{pmatrix}
= \begin{pmatrix} 2 & 1 \\ 6 & -4 \end{pmatrix}
$$
$\text{tr}(BA) = 2 + (-4) = -2$. They match. $\checkmark$

**M2:**
Characteristic polynomial: $\det\begin{pmatrix} 1-\lambda & 2 \\ 0 & -1-\lambda \end{pmatrix} = (1-\lambda)(-1-\lambda) = -(\lambda-1)(\lambda+1) = -\lambda^2 + 1$.

Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = -1$.

$\text{tr}(A) = 1 + (-1) = 0$. $\lambda_1 + \lambda_2 = 1 + (-1) = 0 \quad\checkmark$

**M3:**
$$
A^T A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}
= \begin{pmatrix} 2 & 2 & 2 \\ 2 & 2 & 2 \\ 2 & 2 & 2 \end{pmatrix}
$$
$\text{tr}(A^T A) = 2 + 2 + 2 = 6$.

$\|A\|_F = \sqrt{6}$.

Direct check: $\sqrt{1^2+1^2+1^2+1^2+1^2+1^2} = \sqrt{6} \quad\checkmark$

**M4:**
Let eigenvalues be $\lambda_1, \lambda_2$. We know $\lambda_1 + \lambda_2 = \text{tr}(A) = 4$, and $\lambda_1 \lambda_2 = \det(A) = 3$.

Solving: $\lambda_1, \lambda_2$ are roots of $\lambda^2 - 4\lambda + 3 = 0$, so $\lambda_1 = 1$, $\lambda_2 = 3$ (or vice versa).

**M5:**
If $\text{tr}(A) = 0$, then $\sum_{i=1}^n \lambda_i = 0$. The eigenvalues sum to zero.

### Hard Solutions

**H1:**
If $A = 0$, then $A^T A = 0$, so $\text{tr}(A^T A) = 0$.

Conversely, suppose $\text{tr}(A^T A) = 0$. Then $\sum_{i,j} a_{ij}^2 = 0$. Since each term $a_{ij}^2 \geq 0$, all $a_{ij}$ must be 0, so $A = 0$.

**H2:**
Since $A$ is symmetric, it can be orthogonally diagonalised. Let $\lambda_1, \dots, \lambda_n$ be its eigenvalues. Then $\text{tr}(A) = \sum \lambda_i$ and $\text{tr}(A^2) = \sum \lambda_i^2$.

By the Cauchy-Schwarz inequality (or QM-AM inequality), $(\sum \lambda_i)^2 \leq n \sum \lambda_i^2$, so $\sum \lambda_i^2 \geq \frac{(\sum \lambda_i)^2}{n}$.

Equality holds when all $\lambda_i$ are equal, i.e., $A = cI$ for some scalar $c$.

**H3:**
$(A+B)^T (A+B) = (A^T + B^T)(A + B) = A^T A + A^T B + B^T A + B^T B$.

Taking the trace: $\text{tr}(A^T A + A^T B + B^T A + B^T B)$.

By linearity: $\text{tr}(A^T A) + \text{tr}(A^T B) + \text{tr}(B^T A) + \text{tr}(B^T B)$.

Note that $\text{tr}(B^T A) = \text{tr}((B^T A)^T) = \text{tr}(A^T B)$. So $\text{tr}(A^T B) + \text{tr}(B^T A) = 2\text{tr}(A^T B)$.

Therefore $\text{tr}((A+B)^T (A+B)) = \text{tr}(A^T A) + 2\text{tr}(A^T B) + \text{tr}(B^T B)$.

## Related Concepts

- **Determinant** — Another similarity invariant of square matrices.
- **Eigenvalues** — The trace equals the sum of eigenvalues.
- **Frobenius Norm** — Defined via the trace.
- **Covariance Matrix** — Its trace is total variance.
- **Diagonal Matrix** — The trace of a diagonal matrix is trivial to compute.

## Next Concepts

- **Orthogonal Matrix** (MATH-030) — Orthogonal matrices preserve the trace under similarity.
- **Matrix Exponential** — The trace of $e^A$ relates to eigenvalues.
- **Singular Value Decomposition** — The trace of $A^T A$ gives the sum of squared singular values.
- **Principal Component Analysis** — Uses the trace for variance explained.

## Summary

The trace is the simplest invariant of a square matrix: just the sum of its diagonal entries. Despite its simplicity, it is linear, invariant under cyclic permutations and similarity transformations, and equals the sum of eigenvalues. The trace underpins the Frobenius norm, total variance in PCA, and countless formulas in matrix calculus and machine learning.

## Key Takeaways

- $\text{tr}(A) = \sum_i a_{ii}$ — the sum of diagonal entries.
- The trace is linear: $\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$ and $\text{tr}(cA) = c\,\text{tr}(A)$.
- Cyclic property: $\text{tr}(AB) = \text{tr}(BA)$ (but not arbitrary permutations).
- Similarity invariance makes the trace coordinate-independent.
- $\text{tr}(A) = \sum \lambda_i$ — the sum of eigenvalues.
- The Frobenius norm $\|A\|_F = \sqrt{\text{tr}(A^T A)}$ is the most common matrix norm in ML.
- In PCA, the trace of the covariance matrix gives the total variance.
- The trace appears in loss functions, regularisation, and matrix calculus throughout machine learning.
