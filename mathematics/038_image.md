# Concept: Image (Column Space)

## Concept ID

MATH-038

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Define the image (column space) of a matrix and a linear transformation.
- Compute a basis for the image by identifying pivot columns after row reduction.
- Relate the dimension of the image (the rank) to the number of pivot columns.
- Determine whether a linear transformation is surjective (onto) by comparing the image to the codomain.
- Explain the geometric meaning of the image as the set of all reachable outputs.
- Connect the rank of a feature matrix to the expressiveness of a linear model in regression and classification.
- Analyse the reachable representations of a linear layer in a neural network via its column space.

## Prerequisites

- Matrix operations and multiplication.
- Systems of linear equations and Gaussian elimination.
- Row reduction, pivot positions, reduced row echelon form (RREF).
- Vector spaces and subspaces: span, linear independence, basis, dimension.
- The kernel (nullspace) of a matrix (MATH-037).
- The rank–nullity theorem.
- Linear transformations and their matrix representations.

## Definition

Let $A \in \mathbb{R}^{m \times n}$ be a real matrix with $m$ rows and $n$ columns. The **image** (also called the **column space** or **range**) of $A$ is the set of all vectors $\mathbf{b} \in \mathbb{R}^{m}$ for which there exists some $\mathbf{x} \in \mathbb{R}^{n}$ such that $A\mathbf{x} = \mathbf{b}$.

\[
\operatorname{im}(A) \;=\; \{\,A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^{n}\,\} \;=\; \operatorname{col}(A).
\]

If $T : \mathbb{R}^{n} \to \mathbb{R}^{m}$ is the linear transformation $T(\mathbf{x}) = A\mathbf{x}$, then $\operatorname{im}(T) = \operatorname{im}(A)$. The **rank** of $A$, denoted $\operatorname{rank}(A)$, is the dimension of the image:

\[
\operatorname{rank}(A) \;=\; \dim\bigl(\operatorname{im}(A)\bigr).
\]

Equivalently, the column space is the span of the columns of $A$:

\[
\operatorname{col}(A) \;=\; \operatorname{span}\{\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n\},
\]

where $\mathbf{a}_j \in \mathbb{R}^{m}$ is the $j$-th column of $A$.

## Intuition

Think of a linear transformation as a machine with an input dial (the vector $\mathbf{x}$) and an output display (the vector $\mathbf{b}$). The image is the set of all possible readings on the output display — every output the machine is capable of producing. If the image fills the entire codomain $\mathbb{R}^{m}$, the machine can reach every possible output. If the image is a smaller subspace (say, a plane inside $\mathbb{R}^{3}$), there are some outputs the machine can never produce, no matter how you turn the dials.

Geometrically, if you have a $2 \times 3$ matrix $A$, its columns are three vectors in $\mathbb{R}^{2}$. The image is the span of these three vectors — typically the entire plane $\mathbb{R}^{2}$ if at least two columns are linearly independent, or a line if all columns are multiples of each other, or the origin if all columns are zero.

## Why This Concept Matters

The image is central to understanding the **output behaviour** of linear systems:

- **Surjectivity**: $A$ is surjective (onto) iff $\operatorname{im}(A) = \mathbb{R}^{m}$.
- **Solvability**: The equation $A\mathbf{x} = \mathbf{b}$ has a solution iff $\mathbf{b} \in \operatorname{im}(A)$.
- **Rank**: The dimension of the image (the rank) measures the "richness" or "expressive power" of the linear map. A full-rank matrix preserves maximum information; a low-rank matrix compresses inputs into a lower-dimensional output space.
- **Fundamental theorem**: The image and the kernel of $A^{\mathsf{T}}$ (the left nullspace) are orthogonal complements in $\mathbb{R}^{m}$, giving a complete decomposition of the codomain.

## Historical Background

The concept of the image of a linear map traces back to the early development of matrix theory in the 19th century. **Augustin-Louis Cauchy** (1789–1857) and **James Joseph Sylvester** (1814–1897) studied determinants and the rank of matrices, recognising that the number of independent rows or columns — the rank — measures the "true size" of a linear system. Sylvester coined the term "matrix" in 1850 and developed much of the early theory of canonical forms.

**Georg Frobenius** (1849–1917) systematised the theory of rank and nullity, proving fundamental results about the relationship between the row space and column space. **Élie Cartan** (1869–1951) extended these ideas to differential forms and exterior algebra, where the image of a linear map (now between abstract vector spaces) became a cornerstone of modern differential geometry.

The modern terminology "image" (or "range") for the set of all outputs was standardised in the mid-20th century through the influence of the Bourbaki group and textbooks by **Paul Halmos** and **Serge Lang**. The term "column space" emerged naturally as the span of the columns of a matrix when matrices became the dominant computational tool.

## Real World Examples

### 1. Computer Graphics — Viewing Frustum Transformations

A $4 \times 4$ projection matrix in computer graphics maps 3D points (in homogeneous coordinates) to 2D screen coordinates. The image of this matrix is the set of all points that can appear on the screen. Points outside the image (behind the camera, or outside the frustum) are clipped because they cannot be produced by the transformation.

### 2. Linear Regression — The Hat Matrix

In ordinary least squares, the hat matrix $H = X(X^{\mathsf{T}} X)^{-1} X^{\mathsf{T}}$ projects the response vector $\mathbf{y}$ onto the column space of the design matrix $X$. The image of $H$ is exactly $\operatorname{col}(X)$, and the fitted values $\hat{\mathbf{y}} = H\mathbf{y}$ always lie in this column space. The rank of $X$ determines the number of degrees of freedom in the model.

### 3. Signal Processing — FIR Filters

A finite impulse response (FIR) filter computes a linear combination of input samples: $y[n] = \sum_{k=0}^{M} h_k x[n-k]$. This can be written as a matrix multiplication. The image of the filter matrix describes all possible output signals the filter can produce, given arbitrary inputs.

### 4. Structural Engineering — Load-Bearing Capacity

In a pin-jointed frame, the equilibrium matrix $A$ relates member forces to applied loads. The image of $A$ is the set of all external load vectors that can be sustained by the structure. Loads outside the image correspond to loads that cause the structure to collapse or require infinite member forces. This is directly related to the concept of kinematic determinacy.

### 5. Control Theory — Reachable States

For a linear control system $\dot{\mathbf{x}} = F\mathbf{x} + G\mathbf{u}$, the controllability matrix $\mathcal{C} = [G \; FG \; F^2G \; \dots \; F^{n-1}G]$ determines which states are reachable from zero. The image of $\mathcal{C}$ is precisely the set of reachable states — the reachable subspace. A system is fully controllable iff the image of $\mathcal{C}$ is the entire state space $\mathbb{R}^{n}$.

## AI/ML Relevance

### 1. Linear Layers in Neural Networks

Every fully-connected layer $W \in \mathbb{R}^{m \times n}$ maps $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$. The image $\operatorname{im}(W)$ is the set of all possible output vectors the layer can produce. If $\operatorname{rank}(W) < m$, the layer's outputs are confined to a lower-dimensional subspace of $\mathbb{R}^{m}$. This is called a **representation bottleneck** — the layer cannot produce all possible output patterns, which can either be a limitation (loss of expressivity) or a desired regularisation effect (forcing the network to learn compact representations).

### 2. Rank of Feature Matrices in Linear Regression

Given a design matrix $X \in \mathbb{R}^{N \times d}$ ($N$ samples, $d$ features), the rank of $X$ determines whether the normal equations $X^{\mathsf{T}} X \boldsymbol{\beta} = X^{\mathsf{T}} \mathbf{y}$ have a unique solution. If $\operatorname{rank}(X) = d$ (full column rank), the least-squares solution is unique. If $\operatorname{rank}(X) < d$, there are infinitely many solutions, and regularisation (ridge, LASSO) is needed to select one. The image of $X$ is the subspace of $\mathbb{R}^{N}$ containing all possible fitted values $\hat{\mathbf{y}}$.

### 3. Reachable Representations in Representation Learning

In deep learning, the transformations between layers define a sequence of subspaces. The image of the encoder maps input data into a latent space. The rank of the encoder's weight matrix (or the effective rank after non-linear activation) determines the dimension of the latent representation manifold. Low-rank encoders produce lower-dimensional latent spaces, which is precisely what autoencoders aim to learn.

### 4. Generalisation and Effective Rank

The **effective rank** of a weight matrix (based on its singular value spectrum) is often used to measure the complexity of a neural network. Networks with lower effective rank tend to generalise better because they implicitly constrain the hypothesis space. The image of each layer's weight matrix defines the subspace of activations reachable at that layer, which directly controls the model's capacity.

### 5. PCA and Dimensionality Reduction

Principal Component Analysis computes the best rank-$k$ approximation to a data matrix $X$ by projecting onto the subspace spanned by the top $k$ principal components. The **column space** of the truncated SVD approximation $X_k = U_k \Sigma_k V_k^{\mathsf{T}}$ is the $k$-dimensional subspace that captures maximum variance. This is exactly the column space of $U_k$, which is the span of the first $k$ left singular vectors.

### 6. Matrix Factorisation and Recommender Systems

In collaborative filtering, the user–item interaction matrix $R$ is approximated as a product of two low-rank matrices: $R \approx UV^{\mathsf{T}}$. The image of $U$ (the user embedding matrix) is a $k$-dimensional subspace representing the latent user features. The rank $k$ directly corresponds to the number of latent factors in the model.

## Mathematical Explanation

Let $A \in \mathbb{R}^{m \times n}$ with columns $\mathbf{a}_1, \dots, \mathbf{a}_n \in \mathbb{R}^{m}$. The set of all linear combinations of these columns is:

\[
\operatorname{im}(A) = \bigl\{ x_1\mathbf{a}_1 + x_2\mathbf{a}_2 + \cdots + x_n\mathbf{a}_n \;\big|\; x_1, \dots, x_n \in \mathbb{R} \bigr\}.
\]

This is a subspace of $\mathbb{R}^{m}$ because it is the span of a finite set of vectors.

To compute a basis for $\operatorname{im}(A)$:

1. Row-reduce $A$ to its reduced row echelon form (RREF).
2. Identify the pivot columns (the columns of the RREF that contain leading $1$s).
3. The **original columns** of $A$ corresponding to the pivot positions form a basis for the column space.

**Important**: Do not take the columns of the RREF itself as the basis — row operations change the column space. Only the row space is preserved under row operations. The column space must use the original columns.

The dimension of $\operatorname{im}(A)$ is the rank of $A$, which equals the number of pivot columns. By the rank–nullity theorem:

\[
\operatorname{rank}(A) + \operatorname{nullity}(A) = n.
\]

A linear transformation $T(\mathbf{x}) = A\mathbf{x}$ is **surjective** (onto) iff $\operatorname{im}(A) = \mathbb{R}^{m}$, which requires $\operatorname{rank}(A) = m$. This is only possible if $m \leq n$ (at least as many columns as rows).

The **fundamental theorem of linear algebra** (also called the Fredholm alternative in finite dimensions) gives:

\[
\mathbb{R}^{m} = \operatorname{im}(A) \oplus \ker(A^{\mathsf{T}}),
\]

where $\oplus$ denotes an orthogonal direct sum. That is, the column space and the left nullspace are orthogonal complements. This means the codomain can be split into two orthogonal pieces: the reachable outputs (image) and the unreachable directions (left nullspace).

## Formula(s)

\[
\operatorname{im}(A) = \{ A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^{n} \} \subseteq \mathbb{R}^{m}
\]

\[
\operatorname{im}(A) = \operatorname{span}\{\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n\}
\]

\[
\operatorname{rank}(A) = \dim\bigl(\operatorname{im}(A)\bigr)
\]

\[
\operatorname{rank}(A) = \operatorname{rank}(A^{\mathsf{T}})
\]

\[
\mathbb{R}^{m} = \operatorname{im}(A) \oplus \ker(A^{\mathsf{T}}) \qquad \text{(orthogonal decomposition)}
\]

\[
\operatorname{im}(A) = \mathbb{R}^{m} \iff A \text{ is surjective} \iff \operatorname{rank}(A) = m
\]

## Properties

1. **Subspace**: $\operatorname{im}(A)$ is always a subspace of $\mathbb{R}^{m}$ (the codomain).

2. **Span of columns**: The column space equals the span of the columns of $A$. Similarly, the row space equals $\operatorname{im}(A^{\mathsf{T}})$.

3. **Rank equals column rank**: $\operatorname{rank}(A) = \dim(\operatorname{col}(A))$. This also equals $\dim(\operatorname{row}(A))$ (the row rank).

4. **Surjectivity**: $T(\mathbf{x}) = A\mathbf{x}$ is surjective (onto) iff $\operatorname{im}(A) = \mathbb{R}^{m}$, which requires $\operatorname{rank}(A) = m$.

5. **Image of a product**: $\operatorname{im}(AB) \subseteq \operatorname{im}(A)$. The image of a product is a subspace of the image of the left factor.

6. **Invariance under column operations**: Column operations (adding a multiple of one column to another, swapping columns) preserve the column space.

7. **Image and kernel of the transpose**: $\ker(A^{\mathsf{T}})$ is the orthogonal complement of $\operatorname{im}(A)$. That is, $\mathbf{y} \in \ker(A^{\mathsf{T}})$ iff $\mathbf{y}^{\mathsf{T}} \mathbf{v} = 0$ for all $\mathbf{v} \in \operatorname{im}(A)$.

8. **Full column rank**: If $\operatorname{rank}(A) = n$ (full column rank), then the columns are linearly independent, and $A^{\mathsf{T}} A$ is invertible.

9. **Full row rank**: If $\operatorname{rank}(A) = m$ (full row rank), then $A$ is surjective, and $A A^{\mathsf{T}}$ is invertible.

10. **Low-rank approximation**: For any matrix $A$, the best rank-$k$ approximation in the Frobenius norm is given by truncating the SVD to the top $k$ singular values; the image of the approximation is the span of the first $k$ left singular vectors.

## Step-by-Step Worked Examples

### Example 1: Finding the image of a $3 \times 3$ matrix

Find a basis for $\operatorname{im}(A)$ where $A = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 1 \\ 1 & 2 & 0 \end{bmatrix}$.

**Step 1**: Row-reduce $A$ to RREF.

\[
\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 1 \\ 1 & 2 & 0 \end{bmatrix}
\xrightarrow{R_2 \leftarrow R_2 - 2R_1}
\begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & -1 \\ 1 & 2 & 0 \end{bmatrix}
\]

\[
\xrightarrow{R_3 \leftarrow R_3 - R_1}
\begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & -1 \\ 0 & 0 & -1 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_2}
\begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & -1 \\ 0 & 0 & 0 \end{bmatrix}
\]

\[
\xrightarrow{R_2 \leftarrow -R_2}
\begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}
\xrightarrow{R_1 \leftarrow R_1 - R_2}
\begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]

**Step 2**: Identify pivot columns. The pivots are in columns **1** and **3**.

**Step 3**: Take the original columns of $A$ at index 1 and 3.

\[
\mathbf{a}_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \quad
\mathbf{a}_3 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}.
\]

**Step 4**: These form a basis for $\operatorname{im}(A)$.

\[
\operatorname{im}(A) = \operatorname{span}\left\{
\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix},
\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
\right\}.
\]

Rank: $2$. Verify: $2 + \operatorname{nullity} = 3 \implies \operatorname{nullity}(A) = 1$.

---

### Example 2: Surjectivity (full row rank)

Determine whether $A = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}$ is surjective.

**Step 1**: $A \in \mathbb{R}^{2 \times 3}$, so the codomain is $\mathbb{R}^{2}$.

**Step 2**: Row-reduce to RREF.

\[
\begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}
\xrightarrow{R_1 \leftarrow R_1 - R_2}
\begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \end{bmatrix}.
\]

**Step 3**: Pivot columns are 1 and 2. Rank = 2.

**Step 4**: Since $\operatorname{rank}(A) = 2 = m$, $\operatorname{im}(A) = \mathbb{R}^{2}$, so $A$ is surjective.

Check: The original columns are $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$.
Their span is clearly all of $\mathbb{R}^{2}$ (the first two are linearly independent).

---

### Example 3: Image of a $2 \times 2$ rank-1 matrix

Find $\operatorname{im}(A)$ for $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$.

**Step 1**: Row-reduce.

\[
\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
\xrightarrow{R_2 \leftarrow R_2 - 2R_1}
\begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}.
\]

**Step 2**: Pivot in column 1 only. Rank = 1.

**Step 3**: Original column 1: $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$.

\[
\operatorname{im}(A) = \operatorname{span}\left\{ \begin{bmatrix} 1 \\ 2 \end{bmatrix} \right\}.
\]

Geometrically, this is the line $y = 2x$ in $\mathbb{R}^{2}$. No matter what $\mathbf{x}$ we choose, $A\mathbf{x}$ always lands on this line. So $A$ is not surjective (the image is a 1D line, not the full 2D plane).

**Verification**: For any $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$, $A\mathbf{x} = \begin{bmatrix} x_1 + 2x_2 \\ 2x_1 + 4x_2 \end{bmatrix} = (x_1 + 2x_2)\begin{bmatrix} 1 \\ 2 \end{bmatrix}$, confirming the output is always a multiple of $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$.

---

### Example 4: Image of a product

Let $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$ and $B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$. Find $\operatorname{im}(AB)$ and $\operatorname{im}(A)$.

**Step 1**: Compute $AB = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$.

**Step 2**: $\operatorname{im}(AB)$ is the span of $\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$ — the $xy$-plane in $\mathbb{R}^{3}$.

**Step 3**: $\operatorname{im}(A)$ is exactly the same set (the first two columns of $A$).

This illustrates that $\operatorname{im}(AB) = \operatorname{im}(A)$ when $B$ is full row rank (here $B = I_2$). In general, $\operatorname{im}(AB) \subseteq \operatorname{im}(A)$.

## Visual Interpretation

- **$2 \times 2$ full-rank matrix** ($\operatorname{rank} = 2$): The image is all of $\mathbb{R}^{2}$. The transformation can reach any point in the plane. This corresponds to an invertible transformation (a bijection).

- **$2 \times 2$ rank-1 matrix**: The image is a single line through the origin. All outputs are confined to this line. The transformation squashes the entire 2D plane onto a 1D line. This is the geometric meaning of a non-invertible, singular matrix.

- **$3 \times 2$ matrix with rank 2**: Two columns in $\mathbb{R}^{3}$ typically span a plane through the origin. The image is that plane. Outputs that do not lie in this plane are unreachable — if the system of equations $A\mathbf{x} = \mathbf{b}$ has a $\mathbf{b}$ off this plane, no solution exists.

- **$2 \times 3$ matrix with rank 2**: Three vectors in $\mathbb{R}^{2}$ either span the whole plane (if any two are independent) or a line (if all are colinear). The image is either $\mathbb{R}^{2}$ (surjective) or a line.

- **Zero matrix**: $\operatorname{im}(\mathbf{0}) = \{\mathbf{0}\}$, a zero-dimensional subspace.

Think of the image as the "shadow" of the transformation: no matter what $\mathbf{x}$ you choose, the output always falls within this shadow.

## Common Mistakes

1. **Using the columns of the RREF as a basis for the column space**: Row operations do **not** preserve the column space. Always take the **original columns** of $A$ corresponding to pivot positions.

2. **Confusing $\operatorname{im}(A)$ (columns) with $\operatorname{row}(A)$ (rows)**: The image lives in $\mathbb{R}^{m}$, the row space in $\mathbb{R}^{n}$. They have the same dimension (the rank) but are subspaces of different ambient spaces.

3. **Thinking $\operatorname{rank}(A) = m$ always**: The rank cannot exceed $\min(m, n)$. A $3 \times 2$ matrix can have rank at most 2, so $\operatorname{im}(A) \subsetneq \mathbb{R}^{3}$ always — it can never be surjective.

4. **Confusing surjectivity with injectivity**: Surjectivity is about reaching every output ($\operatorname{im}(A) = \mathbb{R}^{m}$), while injectivity is about unique mapping ($\ker(A) = \{\mathbf{0}\}$). A square matrix can be surjective, injective, both, or neither.

5. **Assuming $\operatorname{im}(AB) = \operatorname{im}(A)$**: In general $\operatorname{im}(AB) \subseteq \operatorname{im}(A)$, but equality holds only under special conditions (e.g., $B$ has full row rank).

6. **Thinking $\operatorname{rank}(A) = \operatorname{rank}(A^{\mathsf{T}})$ is trivial**: It is actually non-trivial and remarkable — the column rank always equals the row rank, a fact that is not obvious from definitions.

7. **Assuming that the number of non-zero rows in RREF equals the number of pivot columns in the original**: While the count is the same (the rank), the specific columns identified as pivot columns in the RREF correspond to original columns whose positions (indices) should be used.

8. **Forgetting that $\operatorname{im}(A) = \mathbb{R}^{m}$ requires $\operatorname{rank}(A) = m$**: Many students mistakenly think any square matrix is automatically surjective. Only invertible square matrices are surjective. Any square matrix with determinant zero has rank $< n$ and is not surjective.

## Interview Questions

### Beginner

1. **Q**: What is the image (column space) of a matrix?
   **A**: The image of $A \in \mathbb{R}^{m \times n}$ is the set of all vectors $\mathbf{b} \in \mathbb{R}^{m}$ such that $A\mathbf{x} = \mathbf{b}$ for some $\mathbf{x} \in \mathbb{R}^{n}$. Equivalently, it is the span of the columns of $A$.

2. **Q**: How do you find a basis for the column space?
   **A**: Row-reduce $A$ to RREF, identify the pivot columns, and take the corresponding **original columns** of $A$.

3. **Q**: What is the rank of a matrix?
   **A**: The rank is the dimension of the image (column space). It equals the number of pivot columns in the RREF.

4. **Q**: When is a matrix $A \in \mathbb{R}^{m \times n}$ surjective?
   **A**: $A$ is surjective (onto) iff $\operatorname{rank}(A) = m$, i.e., the image fills the entire codomain $\mathbb{R}^{m}$.

5. **Q**: Can a $3 \times 2$ matrix be surjective?
   **A**: No. The rank of a $3 \times 2$ matrix is at most $2$, but the codomain has dimension $3$, so $\operatorname{im}(A) \subsetneq \mathbb{R}^{3}$.

### Intermediate

1. **Q**: Prove that $\operatorname{rank}(A) = \operatorname{rank}(A^{\mathsf{T}})$.
   **A**: The rank of $A$ is the dimension of the column space; the rank of $A^{\mathsf{T}}$ is the dimension of the row space of $A$. Row operations preserve the row space and transform the matrix into RREF, where the number of non-zero rows (dimension of row space) equals the number of pivot columns (dimension of column space). Hence the two dimensions are equal.

2. **Q**: Explain the relationship between $\operatorname{im}(A)$ and $\ker(A^{\mathsf{T}})$.
   **A**: $\ker(A^{\mathsf{T}})$ is the orthogonal complement of $\operatorname{im}(A)$ in $\mathbb{R}^{m}$. That is, $\mathbf{y} \in \ker(A^{\mathsf{T}})$ iff $\mathbf{y}^{\mathsf{T}} \mathbf{v} = 0$ for all $\mathbf{v} \in \operatorname{im}(A)$. This gives the orthogonal decomposition $\mathbb{R}^{m} = \operatorname{im}(A) \oplus \ker(A^{\mathsf{T}})$.

3. **Q**: If $A$ is $m \times n$ with rank $r$, what are the possible dimensions of $\operatorname{im}(A)$ and $\ker(A)$?
   **A**: $\dim(\operatorname{im}(A)) = r$, where $0 \leq r \leq \min(m, n)$. By rank–nullity, $\dim(\ker(A)) = n - r$.

4. **Q**: Given $A\mathbf{x} = \mathbf{b}$, what is the condition for the existence of a solution?
   **A**: A solution exists iff $\mathbf{b} \in \operatorname{im}(A)$. Equivalently, $\mathbf{b}$ must be orthogonal to every vector in $\ker(A^{\mathsf{T}})$.

5. **Q**: How does the rank of the design matrix $X$ affect linear regression?
   **A**: If $X$ has full column rank ($\operatorname{rank}(X) = d$), the normal equations $X^{\mathsf{T}} X \boldsymbol{\beta} = X^{\mathsf{T}} \mathbf{y}$ have a unique solution. If $X$ is rank-deficient ($\operatorname{rank}(X) < d$), there are infinitely many solutions, and regularisation is needed.

### Advanced

1. **Q**: In a deep neural network, the weight matrix $W^{(l)}$ at layer $l$ maps $\mathbb{R}^{d_{l-1}}$ to $\mathbb{R}^{d_{l}}$. How does $\operatorname{im}(W^{(l)})$ change after applying a non-linear activation function like ReLU?
   **A**: The non-linear activation function restricts the reachable outputs. After ReLU, the output is the cone $\{\max(0, W^{(l)}\mathbf{x}) \mid \mathbf{x} \in \mathbb{R}^{d_{l-1}}\}$, which is the intersection of $\operatorname{im}(W^{(l)})$ with the non-negative orthant, further modified by the piecewise nature of ReLU. The effective rank of the layer's output distribution can be estimated via the singular value spectrum of $W^{(l)}$, but the non-linearity can increase the effective rank by bending the subspace into a manifold.

2. **Q**: The **Johnson–Lindenstrauss lemma** states that $N$ points in $\mathbb{R}^{d}$ can be embedded into $\mathbb{R}^{k}$ with $k = O(\log N / \varepsilon^2)$ while approximately preserving pairwise distances. What does this imply about the image of a random projection matrix?
   **A**: A random matrix $R \in \mathbb{R}^{k \times d}$ with independent Gaussian entries has (with high probability) $\operatorname{im}(R) = \mathbb{R}^{k}$ (full rank), but more importantly, the map is nearly an isometry on the set of points when restricted to their span. The image under $R$ of the point cloud is a faithful lower-dimensional representation, and the fact that $k$ can be logarithmic in $N$ shows that high-dimensional data often has low "intrinsic dimension."

3. **Q**: In the context of the SVD, $A = U \Sigma V^{\mathsf{T}}$, describe $\operatorname{im}(A)$, $\operatorname{im}(A^{\mathsf{T}})$, $\ker(A)$, and $\ker(A^{\mathsf{T}})$ in terms of $U$, $V$, and $\Sigma$.
   **A**: If $A \in \mathbb{R}^{m \times n}$ has rank $r$, write $U = [U_r \; U_{m-r}]$ and $V = [V_r \; V_{n-r}]$. Then:
   - $\operatorname{im}(A) = \operatorname{span}(U_r)$ — the first $r$ left singular vectors.
   - $\ker(A^{\mathsf{T}}) = \operatorname{span}(U_{m-r})$ — the last $m-r$ left singular vectors, which are orthogonal to $\operatorname{im}(A)$.
   - $\operatorname{im}(A^{\mathsf{T}}) = \operatorname{span}(V_r)$ — the first $r$ right singular vectors.
   - $\ker(A) = \operatorname{span}(V_{n-r})$ — the last $n-r$ right singular vectors.
   This gives an explicit orthonormal basis for all four fundamental subspaces via the SVD.

## Practice Problems

### Easy

1. Find a basis for $\operatorname{im}(A)$ and compute the rank: $A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$.

2. Find a basis for $\operatorname{im}(A)$ and compute the rank: $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$.

3. Determine whether $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ is surjective.

4. For $A = \begin{bmatrix} 1 & -1 \\ -2 & 2 \end{bmatrix}$, find $\operatorname{im}(A)$ and describe it geometrically.

5. If $A$ is $m \times n$ and $\operatorname{rank}(A) = 3$, what is $\dim(\operatorname{im}(A))$? What can you say about $m$ and $n$?

### Medium

1. Find a basis for $\operatorname{im}(A)$ where $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 2 & 3 \end{bmatrix}$. What is the rank?

2. Prove that $\operatorname{im}(AB) \subseteq \operatorname{im}(A)$ for any matrices $A$ and $B$ where the product is defined.

3. For $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$, find $\operatorname{im}(A)$, $\ker(A^{\mathsf{T}})$, and verify the orthogonal decomposition $\mathbb{R}^{3} = \operatorname{im}(A) \oplus \ker(A^{\mathsf{T}})$.

4. Let $A \in \mathbb{R}^{m \times n}$. Show that $A$ is surjective iff $A A^{\mathsf{T}}$ is invertible.

5. For $A = \begin{bmatrix} 1 & 2 & 0 \\ 2 & 1 & 3 \\ 0 & 0 & 1 \end{bmatrix}$, find bases for $\operatorname{im}(A)$ and $\operatorname{im}(A^{\mathsf{T}})$, and verify that $\dim(\operatorname{im}(A)) = \dim(\operatorname{im}(A^{\mathsf{T}}))$.

### Hard

1. Let $A \in \mathbb{R}^{m \times n}$ with $\operatorname{rank}(A) = r$. Prove that there exists a matrix $B \in \mathbb{R}^{m \times r}$ and $C \in \mathbb{R}^{r \times n}$ such that $A = BC$ and both $B$ and $C$ have rank $r$ (a **rank factorisation**). What is $\operatorname{im}(B)$?

2. Consider a linear neural network with two layers: $f(\mathbf{x}) = W_2 W_1 \mathbf{x}$ where $W_1 \in \mathbb{R}^{d_1 \times d_0}$, $W_2 \in \mathbb{R}^{d_2 \times d_1}$. Show that $\operatorname{rank}(W_2 W_1) \leq \min(\operatorname{rank}(W_1), \operatorname{rank}(W_2))$. What is the maximum possible rank of $f$? Under what conditions does the rank decrease?

3. For a symmetric positive semi-definite matrix $S$, prove that $\operatorname{im}(S) = \operatorname{im}(S^{1/2})$ and that $\ker(S) = \ker(S^{1/2})$. Show that $\mathbb{R}^{n} = \operatorname{im}(S) \oplus \ker(S)$ is an orthogonal decomposition.

## Solutions

### Easy — Solutions

**1.** $A = I_3$ is already in RREF with pivots in columns 1, 2, 3. Rank = 3. Basis: $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ — the standard basis. $\operatorname{im}(A) = \mathbb{R}^{3}$.

**2.** RREF: $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$. Pivot in column 1. Rank = 1. Basis: $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$. $\operatorname{im}(A) = \operatorname{span}\{\begin{bmatrix} 1 \\ 0 \end{bmatrix}\}$.

**3.** $\det(A) = 1\cdot 4 - 2\cdot 3 = -2 \neq 0$, so $A$ is invertible and full rank. $\operatorname{rank}(A) = 2 = m$, so $A$ is surjective.

**4.** $A = \begin{bmatrix} 1 & -1 \\ -2 & 2 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$. Pivot in column 1. Rank = 1. Basis: $\begin{bmatrix} 1 \\ -2 \end{bmatrix}$. Geometrically, $\operatorname{im}(A)$ is the line $y = -2x$ in $\mathbb{R}^{2}$.

**5.** $\dim(\operatorname{im}(A)) = \operatorname{rank}(A) = 3$. Since $\operatorname{rank}(A) \leq \min(m, n)$, we must have $m \geq 3$ and $n \geq 3$.

### Medium — Solutions

**1.** Row-reduce:

\[
\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 2 & 3 \end{bmatrix}
\xrightarrow{R_2 \leftarrow R_2 - 2R_1,\; R_3 \leftarrow R_3 - R_1}
\begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}.
\]

Pivot in column 1 only. Rank = 1. Basis: $\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}$. All columns are multiples of the first column.

**2.** Take any $\mathbf{y} \in \operatorname{im}(AB)$. Then $\mathbf{y} = AB\mathbf{x}$ for some $\mathbf{x}$. Let $\mathbf{z} = B\mathbf{x}$. Then $\mathbf{y} = A\mathbf{z}$, so $\mathbf{y} \in \operatorname{im}(A)$. Hence $\operatorname{im}(AB) \subseteq \operatorname{im}(A)$.

**3.** Row-reduce $A$:

\[
A \xrightarrow{R_3 \leftarrow R_3 - R_1}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_2}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]

Pivots in columns 1, 2. Basis for $\operatorname{im}(A)$: original columns 1 and 2: $\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$, $\begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$.

Now find $\ker(A^{\mathsf{T}})$: $A^{\mathsf{T}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$. Solve $A^{\mathsf{T}} \mathbf{y} = \mathbf{0}$:

\[
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_1}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix}
\xrightarrow{R_3 \leftarrow R_3 - R_2}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}.
\]

Equations: $y_1 + y_3 = 0$, $y_2 + y_3 = 0 \implies y_1 = y_2 = -y_3$. Basis: $\begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}$.

Check orthogonality: $\begin{bmatrix} 1 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix} = 1 + 0 - 1 = 0$, and $\begin{bmatrix} 0 & 1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix} = 0 + 1 - 1 = 0$. So $\operatorname{im}(A) \perp \ker(A^{\mathsf{T}})$, and $2 + 1 = 3 = m$, giving $\mathbb{R}^{3} = \operatorname{im}(A) \oplus \ker(A^{\mathsf{T}})$.

**4.** ($\Rightarrow$) If $A$ is surjective, $\operatorname{rank}(A) = m$, so $A$ has full row rank. Then $A A^{\mathsf{T}} \in \mathbb{R}^{m \times m}$ has $\operatorname{rank}(A A^{\mathsf{T}}) = \operatorname{rank}(A) = m$, so $A A^{\mathsf{T}}$ is invertible. ($\Leftarrow$) If $A A^{\mathsf{T}}$ is invertible, then $\operatorname{rank}(A A^{\mathsf{T}}) = m$. Since $\operatorname{rank}(A A^{\mathsf{T}}) \leq \operatorname{rank}(A) \leq m$, we must have $\operatorname{rank}(A) = m$, so $A$ is surjective.

**5.** Row-reduce $A$:

\[
A = \begin{bmatrix} 1 & 2 & 0 \\ 2 & 1 & 3 \\ 0 & 0 & 1 \end{bmatrix}
\xrightarrow{R_2 \leftarrow R_2 - 2R_1}
\begin{bmatrix} 1 & 2 & 0 \\ 0 & -3 & 3 \\ 0 & 0 & 1 \end{bmatrix}
\xrightarrow{R_2 \leftarrow -R_2/3}
\begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & -1 \\ 0 & 0 & 1 \end{bmatrix}
\xrightarrow{R_2 \leftarrow R_2 + R_3}
\begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
\xrightarrow{R_1 \leftarrow R_1 - 2R_2}
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}.
\]

Rank = 3. Basis for $\operatorname{im}(A)$: all three original columns (since they are the standard basis after accounting for row operations, but we use original columns): $\begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 3 \\ 1 \end{bmatrix}$.

For $\operatorname{im}(A^{\mathsf{T}})$: $A^{\mathsf{T}} = \begin{bmatrix} 1 & 2 & 0 \\ 2 & 1 & 0 \\ 0 & 3 & 1 \end{bmatrix}$. $A^{\mathsf{T}}$ also has rank 3 (since $\operatorname{rank}(A^{\mathsf{T}}) = \operatorname{rank}(A) = 3$). Basis: all three columns of $A^{\mathsf{T}}$: $\begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$.

Both have dimension 3, confirming $\operatorname{rank}(A) = \operatorname{rank}(A^{\mathsf{T}})$.

### Hard — Solutions

**1.** Let $A = U \Sigma V^{\mathsf{T}}$ be the SVD of $A$ with rank $r$. Write $U = [U_r \; U_{m-r}]$, $\Sigma = \begin{bmatrix} \Sigma_r & 0 \\ 0 & 0 \end{bmatrix}$, $V = [V_r \; V_{n-r}]$. Then $A = U_r \Sigma_r V_r^{\mathsf{T}}$. Set $B = U_r \Sigma_r \in \mathbb{R}^{m \times r}$ and $C = V_r^{\mathsf{T}} \in \mathbb{R}^{r \times n}$. Then $A = BC$, $\operatorname{rank}(B) = r$, $\operatorname{rank}(C) = r$. $\operatorname{im}(B) = \operatorname{span}(U_r) = \operatorname{im}(A)$, since $U_r$ spans the column space of $A$.

**2.** $\operatorname{im}(W_2 W_1) \subseteq \operatorname{im}(W_2)$, so $\operatorname{rank}(W_2 W_1) \leq \operatorname{rank}(W_2)$. Also, by considering $W_2 W_1$ as a composition, $\operatorname{rank}(W_2 W_1) \leq \operatorname{rank}(W_1)$ because the image of $W_1$ is at most $r_1$-dimensional, and $W_2$ maps this to at most $r_1$ dimensions. Hence $\operatorname{rank}(W_2 W_1) \leq \min(\operatorname{rank}(W_1), \operatorname{rank}(W_2))$. The maximum possible rank of $f$ is $\min(d_0, d_1, d_2)$. The rank decreases if the image of $W_1$ has non-trivial intersection with the kernel of $W_2$, i.e., if $\operatorname{im}(W_1) \cap \ker(W_2) \neq \{\mathbf{0}\}$.

**3.** Since $S$ is positive semi-definite, it has a unique positive semi-definite square root $S^{1/2}$. Then $S = S^{1/2} S^{1/2}$. By property (Medium 2), $\operatorname{im}(S) \subseteq \operatorname{im}(S^{1/2})$. Conversely, if $\mathbf{y} \in \operatorname{im}(S^{1/2})$, then $\mathbf{y} = S^{1/2} \mathbf{x}$ for some $\mathbf{x}$, so $\mathbf{y} = S^{1/2} \mathbf{x} = S^{1/2} (S^{1/2} S^{1/2})^{-1} S^{1/2} \mathbf{x}$ if we work with pseudoinverses; a cleaner argument: $S^{1/2} = Q \Lambda^{1/2} Q^{\mathsf{T}}$ and $S = Q \Lambda Q^{\mathsf{T}}$, so their images are both $\operatorname{span}$ of the eigenvectors with positive eigenvalues. Hence $\operatorname{im}(S) = \operatorname{im}(S^{1/2})$. Similarly, $\ker(S) = \ker(S^{1/2})$ because $S\mathbf{x} = \mathbf{0} \iff \Lambda Q^{\mathsf{T}} \mathbf{x} = \mathbf{0} \iff \Lambda^{1/2} Q^{\mathsf{T}} \mathbf{x} = \mathbf{0}$. Since $S$ is symmetric, $\operatorname{im}(S) \perp \ker(S)$ (by the property that image and left nullspace are orthogonal, and for symmetric $S$, $\ker(S) = \ker(S^{\mathsf{T}})$). So $\mathbb{R}^{n} = \operatorname{im}(S) \oplus \ker(S)$ orthogonally.

## Related Concepts

- **Kernel (Nullspace)** (MATH-037): The set of inputs mapping to zero; dual to the image through the rank–nullity theorem.
- **Row Space**: The span of the rows of $A$; equal to $\operatorname{im}(A^{\mathsf{T}})$; same dimension as the column space.
- **Left Nullspace**: $\ker(A^{\mathsf{T}})$, the orthogonal complement of $\operatorname{im}(A)$.
- **Rank–Nullity Theorem**: $\operatorname{rank}(A) + \operatorname{nullity}(A) = n$, linking the image and kernel.
- **Fundamental Theorem of Linear Algebra**: Describes the four fundamental subspaces and their orthogonal relationships.
- **Singular Value Decomposition**: A factorisation that reveals orthonormal bases for $\operatorname{im}(A)$, $\operatorname{im}(A^{\mathsf{T}})$, $\ker(A)$, and $\ker(A^{\mathsf{T}})$.
- **Moore–Penrose Pseudoinverse**: The linear map that sends $\mathbf{b}$ to the minimum-norm solution of $A\mathbf{x} = \mathbf{b}$ when $\mathbf{b} \in \operatorname{im}(A)$.

## Next Concepts

- **Left Nullspace** (MATH-039): The orthogonal complement of the column space — what outputs are unreachable.
- **Four Fundamental Subspaces**: A unified treatment of $\operatorname{im}(A)$, $\ker(A)$, $\operatorname{im}(A^{\mathsf{T}})$, and $\ker(A^{\mathsf{T}})$.
- **Singular Value Decomposition**: The practical tool for computing orthonormal bases of all four subspaces.
- **Linear Regression and the Hat Matrix**: Application of column spaces to statistics and data science.
- **Controllability and Observability**: Control-theoretic concepts that directly use the image and kernel of the controllability and observability matrices.

## Summary

The image (column space) of a matrix $A \in \mathbb{R}^{m \times n}$ is the set of all possible outputs $A\mathbf{x}$ as $\mathbf{x}$ ranges over $\mathbb{R}^{n}$. It is a subspace of $\mathbb{R}^{m}$ spanned by the columns of $A$. Its dimension is the rank of $A$, which satisfies $0 \leq \operatorname{rank}(A) \leq \min(m, n)$. The image and the left nullspace ($\ker(A^{\mathsf{T}})$) are orthogonal complements that partition the codomain $\mathbb{R}^{m}$. A matrix is surjective (onto) precisely when its image is the whole codomain, i.e., when $\operatorname{rank}(A) = m$. Computing the image amounts to identifying pivot columns in the RREF and taking the corresponding original columns as a basis. In machine learning, the column space of a weight matrix defines the set of all reachable outputs of a linear layer, and the rank controls the expressive capacity of the model. The concept is fundamental to understanding solvability of linear systems, least-squares problems, representation learning, and dimensionality reduction.

## Key Takeaways

- $\operatorname{im}(A) = \{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^{n}\} = \operatorname{col}(A)$ is a subspace of $\mathbb{R}^{m}$.
- The dimension of the image is the **rank** of $A$.
- A basis for $\operatorname{im}(A)$ is found by taking the original columns of $A$ corresponding to pivot columns in the RREF.
- $A$ is surjective iff $\operatorname{im}(A) = \mathbb{R}^{m}$, i.e., $\operatorname{rank}(A) = m$.
- $\mathbb{R}^{m} = \operatorname{im}(A) \oplus \ker(A^{\mathsf{T}})$ — the codomain splits into reachable and unreachable directions.
- $\operatorname{rank}(A) = \operatorname{rank}(A^{\mathsf{T}})$ — column rank equals row rank.
- $\operatorname{im}(AB) \subseteq \operatorname{im}(A)$ — the image of a product is contained in the image of the left factor.
- In ML, the column space of a weight matrix determines what representations a linear layer can produce; the rank controls the model's expressive capacity.
