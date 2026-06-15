# Concept: Matrix Multiplication

## Concept ID

MATH-023

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define matrix multiplication using the dot product of rows and columns
- Apply the dimension compatibility rule: $m \times n$ times $n \times p$ yields $m \times p$
- Explain why matrix multiplication is not commutative
- Verify associativity, distributivity, and the identity property
- Connect matrix multiplication to neural network forward passes and attention mechanisms
- Distinguish matrix multiplication from element-wise multiplication

## Prerequisites

- Understanding of matrices as rectangular arrays of numbers (MATH-003)
- Matrix addition and subtraction (MATH-021, MATH-022)
- The dot product of two vectors (MATH-016)
- Basic algebraic manipulation

## Definition

**Matrix multiplication** is an operation that takes two matrices and produces a third matrix. Unlike addition and subtraction, it is **not** element-wise. Instead, each entry of the product is computed as the dot product of a row from the first matrix and a column from the second.

If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, their product $C = AB$ is an $m \times p$ matrix where:

$$
c_{ij} = \sum_{k=1}^{n} a_{ik} \, b_{kj}
$$

for $i = 1, 2, \dots, m$ and $j = 1, 2, \dots, p$.

The number of columns in $A$ ($n$) must equal the number of rows in $B$ ($n$). This is the **dimension compatibility rule**. If this condition is not met, the product is undefined.

For example, if:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

then:

$$
AB = \begin{bmatrix}
1\cdot5 + 2\cdot7 & 1\cdot6 + 2\cdot8 \\
3\cdot5 + 4\cdot7 & 3\cdot6 + 4\cdot8
\end{bmatrix}
= \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

## Intuition

Matrix multiplication combines two linear transformations into one. If matrix $A$ represents one transformation (e.g., rotate 90 degrees) and matrix $B$ represents another (e.g., scale by 2), then $AB$ represents applying $B$ first, then $A$ (right to left).

Entry $c_{ij}$ captures how much of row $i$ of $A$ "aligns with" column $j$ of $B$. Think of it as: take row $i$ of $A$ (a list of $n$ numbers), take column $j$ of $B$ (a list of $n$ numbers), multiply them pairwise, and sum. That single number tells you the strength of the connection between input $i$ and output $j$ through all $n$ intermediate pathways.

## Why This Concept Matters

Matrix multiplication is arguably the most important operation in linear algebra and the backbone of modern AI. It is essential for:

- **Neural networks**: Every forward pass is a sequence of matrix multiplications
- **Computer graphics**: Transforming coordinates, rotating, scaling, and projecting objects
- **Solving linear systems**: $Ax = b$ is the canonical problem in scientific computing
- **Data transformations**: PCA, SVD, and other dimensionality reduction techniques all rely on matrix multiplication
- **Quantum mechanics**: State transitions are matrix multiplications
- **Economics**: Input-output models describe how industries interact through matrix products

Without matrix multiplication, deep learning as we know it would not exist.

## Historical Background

Matrix multiplication was first defined by **Arthur Cayley** in 1858 in his "A Memoir on the Theory of Matrices." Cayley recognised that composing linear transformations corresponds to multiplying their matrices. The definition was motivated by the need to represent the composition of two linear functions. If $y = f(x) = Ax$ and $z = g(y) = By$, then $z = g(f(x)) = (BA)x$, and the matrix representing the composed transformation is $BA$ (not $AB$ — note the order!).

The non-commutativity of matrix multiplication was understood from the start — it reflects the fact that applying transformations in different orders produces different results. This was a radical departure from ordinary arithmetic, where multiplication has always been commutative.

## Real World Examples

1. **Population Dynamics**: A transition matrix $T$ describes how populations move between states (urban, suburban, rural) each year. Multiplying $T$ by the current population vector gives next year's distribution.

2. **Cryptography**: Some encryption schemes encode messages as matrices and multiply by a secret key matrix to produce ciphertext. Decryption uses the inverse matrix.

3. **Supply Chain**: An input-output matrix $A$ describes how much of each industry's output is consumed by each other industry. Multiplying $A$ by a demand vector gives total production requirements.

4. **Google PageRank**: The PageRank algorithm computes the eigenvector of a giant link matrix, which involves repeated matrix-vector multiplications until convergence.

5. **Robotics**: A robot arm's end-effector position is computed by multiplying a chain of transformation matrices, each representing a joint rotation or link translation.

## AI/ML Relevance

Matrix multiplication is the computational engine of deep learning.

1. **Forward Pass in Neural Networks**: A fully-connected layer computes $y = Wx + b$. Here $W$ is an $m \times n$ weight matrix, $x$ is an $n$-dimensional input vector (or an $n \times b$ matrix for a batch), and $y$ is the $m$-dimensional output. Each entry $y_j$ is the dot product of row $j$ of $W$ with $x$:
   $$
   y_j = \sum_{k=1}^{n} w_{jk} x_k + b_j
   $$
   Every neuron in the layer is computing one dot product. A network with 1000 neurons per layer and 100 layers performs millions of matrix multiplications per forward pass.

2. **Attention Mechanisms**: In transformers, the attention scores are computed as:
   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
   $$
   Here $Q$ (queries), $K$ (keys), and $V$ (values) are matrices. The product $QK^T$ computes dot products between every query-key pair — a massive matrix multiplication. This is the core operation in GPT, BERT, Llama, and virtually every modern language model.

3. **Convolution as Matrix Multiplication**: A convolution operation can be expressed as matrix multiplication via the **im2col** technique. The input image is expanded into a matrix where each column corresponds to a receptive field, and the convolution kernels form another matrix. Their product produces the output. This is how deep learning frameworks implement convolutions efficiently.

4. **Word Embeddings**: A lookup into an embedding matrix $E \in \mathbb{R}^{V \times d}$ ($V$ vocabulary size, $d$ embedding dimension) followed by multiplication with a weight matrix is how language models represent words.

5. **Principal Component Analysis (PCA)** : The data matrix $X$ is centred and then its covariance matrix $X^T X$ is computed (a matrix multiplication). The eigenvectors of this product give the principal components.

6. **Self-Supervised Learning**: Contrastive learning methods like SimCLR compute similarity matrices as $ZZ^T$ where $Z$ is a matrix of normalised embeddings. This is a matrix multiplication that compares every sample with every other sample.

## Mathematical Explanation

### The Dot Product of a Row and Column

Each entry $c_{ij}$ of $AB$ is the dot product of row $i$ of $A$ and column $j$ of $B$:

$$
c_{ij} = \underbrace{\begin{bmatrix} a_{i1} & a_{i2} & \cdots & a_{in} \end{bmatrix}}_{\text{row } i \text{ of } A}
\cdot
\underbrace{\begin{bmatrix} b_{1j} \\ b_{2j} \\ \vdots \\ b_{nj} \end{bmatrix}}_{\text{column } j \text{ of } B}
= \sum_{k=1}^{n} a_{ik} b_{kj}
$$

### Dimension Compatibility — Why It Matters

The rule is: if $A$ is $m \times n$ and $B$ is $n \times p$, then $AB$ is defined and the result is $m \times p$.

Why must the inner dimensions ($n$) match? Because the dot product requires two vectors of the same length. Row $i$ of $A$ has $n$ entries, and column $j$ of $B$ has $n$ entries. If $A$ had $n$ columns and $B$ had $r \neq n$ rows, the row and column vectors would have different lengths and the dot product would be undefined.

The outer dimensions ($m$ and $p$) determine the shape of the result. You can visualise this as:

$$
(m \times \boxed{n}) \; \text{times} \; (\boxed{n} \times p) = m \times p
$$

### Why Non-Commutativity

$AB$ and $BA$ are generally not equal, for three reasons:

1. **Dimension mismatch**: If $A$ is $2 \times 3$ and $B$ is $3 \times 4$, then $AB$ is $2 \times 4$ but $BA$ is undefined (a $3 \times 4$ times $2 \times 3$ violates the inner dimension rule).

2. **Different result shapes**: Even if both products are defined (both matrices are square), the entries differ because the dot products pair different rows and columns.

3. **Transformations don't commute**: Rotating then scaling is different from scaling then rotating. Since matrices encode transformations, the order matters.

### Relationship to Linear Transformations

If $A$ represents linear transformation $T_A$ and $B$ represents $T_B$, then $AB$ represents $T_A \circ T_B$ (apply $T_B$ first, then $T_A$). This is why $AB$ is read right-to-left in transformation contexts.

## Formula(s)

**General formula:**

$$
(AB)_{ij} = \sum_{k=1}^{n} a_{ik} \, b_{kj}
$$

**$2 \times 2$ case:**

$$
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix}
=
\begin{bmatrix}
a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\
a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22}
\end{bmatrix}
$$

**Matrix-vector multiplication** ($A$ is $m \times n$, $x$ is $n \times 1$):

$$
(Ax)_i = \sum_{k=1}^{n} a_{ik} x_k
$$

## Properties

1. **Not Commutative**: In general, $AB \neq BA$. Even when both products exist (square matrices), they usually differ.

2. **Associative**: $(AB)C = A(BC)$, provided the dimensions match for the products. This allows us to group operations flexibly.

3. **Distributive**: $A(B + C) = AB + AC$ and $(A + B)C = AC + BC$.

4. **Identity Property**: If $A$ is $m \times n$, then $I_m A = A$ and $A I_n = A$, where $I_k$ is the $k \times k$ identity matrix.

5. **Zero Property**: If $A$ is $m \times n$, then $A \cdot 0_{n \times p} = 0_{m \times p}$ and $0_{p \times m} \cdot A = 0_{p \times n}$.

6. **Compatibility with Scalar Multiplication**: $c(AB) = (cA)B = A(cB)$ for any scalar $c$.

7. **Transpose of a Product**: $(AB)^T = B^T A^T$. Note the reversal of order.

8. **No Division**: There is no matrix division. Instead, we multiply by an inverse (when it exists).

## Step-by-Step Worked Examples

### Example 1: Multiplying Two $2 \times 2$ Matrices

**Problem:** Let $A = \begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix}$ and $B = \begin{bmatrix} 1 & 4 \\ -2 & 5 \end{bmatrix}$. Compute $AB$.

**Solution:**

**Step 1:** Check dimensions. $A$ is $2 \times 2$, $B$ is $2 \times 2$. The inner dimensions match (2 = 2). The result will be $2 \times 2$.

**Step 2:** Compute each entry as a dot product:

- $c_{11}$ = row 1 of $A$ $\cdot$ column 1 of $B$: $[2, -1] \cdot [1, -2] = 2(1) + (-1)(-2) = 2 + 2 = 4$
- $c_{12}$ = row 1 of $A$ $\cdot$ column 2 of $B$: $[2, -1] \cdot [4, 5] = 2(4) + (-1)(5) = 8 - 5 = 3$
- $c_{21}$ = row 2 of $A$ $\cdot$ column 1 of $B$: $[0, 3] \cdot [1, -2] = 0(1) + 3(-2) = 0 - 6 = -6$
- $c_{22}$ = row 2 of $A$ $\cdot$ column 2 of $B$: $[0, 3] \cdot [4, 5] = 0(4) + 3(5) = 0 + 15 = 15$

**Step 3:** Assemble:

$$
AB = \begin{bmatrix} 4 & 3 \\ -6 & 15 \end{bmatrix}
$$

### Example 2: Multiplying a $3 \times 2$ and $2 \times 4$ Matrix

**Problem:** Let $A = \begin{bmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 2 & 0 & -1 & 3 \\ 1 & 4 & 2 & -2 \end{bmatrix}$. Compute $AB$.

**Solution:**

**Step 1:** $A$ is $3 \times 2$, $B$ is $2 \times 4$. Inner dimension: $2 = 2$. Result is $3 \times 4$.

**Step 2:** Compute row by row.

Row 1 of $A = [1, 2]$:
- Col 1: $1(2) + 2(1) = 2 + 2 = 4$
- Col 2: $1(0) + 2(4) = 0 + 8 = 8$
- Col 3: $1(-1) + 2(2) = -1 + 4 = 3$
- Col 4: $1(3) + 2(-2) = 3 - 4 = -1$

Row 2 of $A = [0, -1]$:
- Col 1: $0(2) + (-1)(1) = 0 - 1 = -1$
- Col 2: $0(0) + (-1)(4) = 0 - 4 = -4$
- Col 3: $0(-1) + (-1)(2) = 0 - 2 = -2$
- Col 4: $0(3) + (-1)(-2) = 0 + 2 = 2$

Row 3 of $A = [3, 4]$:
- Col 1: $3(2) + 4(1) = 6 + 4 = 10$
- Col 2: $3(0) + 4(4) = 0 + 16 = 16$
- Col 3: $3(-1) + 4(2) = -3 + 8 = 5$
- Col 4: $3(3) + 4(-2) = 9 - 8 = 1$

**Step 3:** Assemble:

$$
AB = \begin{bmatrix}
4 & 8 & 3 & -1 \\
-1 & -4 & -2 & 2 \\
10 & 16 & 5 & 1
\end{bmatrix}
$$

### Example 3: Demonstrating Non-Commutativity

**Problem:** Let $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$. Compute $AB$ and $BA$ and show they are different.

**Solution:**

**Step 1:** Compute $AB$:

$$
AB = \begin{bmatrix} 1(0)+2(1) & 1(1)+2(0) \\ 3(0)+4(1) & 3(1)+4(0) \end{bmatrix}
= \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}
$$

**Step 2:** Compute $BA$:

$$
BA = \begin{bmatrix} 0(1)+1(3) & 0(2)+1(4) \\ 1(1)+0(3) & 1(2)+0(4) \end{bmatrix}
= \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}
$$

**Step 3:** Compare:

$$
AB = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}, \quad
BA = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}
$$

Clearly $AB \neq BA$. The matrix $B$ swaps rows when multiplying on the left ($BA$) and swaps columns when multiplying on the right ($AB$). These are different operations.

### Example 4: Forward Pass in a Neural Network Layer

**Problem:** A neural network layer has weight matrix $W = \begin{bmatrix} 0.5 & -0.2 & 0.1 \\ -0.3 & 0.8 & 0.4 \end{bmatrix}$ and bias $b = [0.1, -0.2]$. For an input vector $x = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}$, compute the pre-activation $z = Wx + b$.

**Solution:**

**Step 1:** Check dimensions. $W$ is $2 \times 3$, $x$ is $3 \times 1$. Inner dimensions match ($3 = 3$). Result $Wx$ will be $2 \times 1$.

**Step 2:** Compute $Wx$:

- Entry 1: $[0.5, -0.2, 0.1] \cdot [2, -1, 3] = 0.5(2) + (-0.2)(-1) + 0.1(3) = 1 + 0.2 + 0.3 = 1.5$
- Entry 2: $[-0.3, 0.8, 0.4] \cdot [2, -1, 3] = -0.3(2) + 0.8(-1) + 0.4(3) = -0.6 - 0.8 + 1.2 = -0.2$

So $Wx = \begin{bmatrix} 1.5 \\ -0.2 \end{bmatrix}$.

**Step 3:** Add bias: $z = Wx + b = \begin{bmatrix} 1.5 + 0.1 \\ -0.2 + (-0.2) \end{bmatrix} = \begin{bmatrix} 1.6 \\ -0.4 \end{bmatrix}$.

**Step 4:** The output $z$ is then passed through an activation function like ReLU: $\text{ReLU}(z) = \begin{bmatrix} 1.6 \\ 0 \end{bmatrix}$.

This single $Wx$ multiplication computes the weighted input to all 2 neurons in the layer from the 3 input features.

### Example 5: Batch Matrix Multiplication

**Problem:** A batch of 2 samples, each with 3 features, is stored as $X = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \end{bmatrix}$ ($2 \times 3$). The weight matrix is $W = \begin{bmatrix} 0.5 & -1 \\ 0.2 & 0.3 \\ -0.5 & 0.7 \end{bmatrix}$ ($3 \times 2$). Compute $XW$.

**Solution:**

**Step 1:** $X$ is $2 \times 3$, $W$ is $3 \times 2$. Inner dimensions match ($3 = 3$). Result is $2 \times 2$.

**Step 2:** Compute each entry:

Sample 1 ($[1, 0, -1]$):
- Output 1: $1(0.5) + 0(0.2) + (-1)(-0.5) = 0.5 + 0 + 0.5 = 1.0$
- Output 2: $1(-1) + 0(0.3) + (-1)(0.7) = -1 + 0 - 0.7 = -1.7$

Sample 2 ($[2, 1, 0]$):
- Output 1: $2(0.5) + 1(0.2) + 0(-0.5) = 1.0 + 0.2 + 0 = 1.2$
- Output 2: $2(-1) + 1(0.3) + 0(0.7) = -2 + 0.3 + 0 = -1.7$

**Step 3:** Assemble:

$$
XW = \begin{bmatrix} 1.0 & -1.7 \\ 1.2 & -1.7 \end{bmatrix}
$$

Each row of the output corresponds to the transformed features for one sample.

## Visual Interpretation

Matrix multiplication can be visualised as a "row times column" dance. For each output entry, a row vector from $A$ slides over a column vector from $B$, multiplying pairs and summing.

If you think of $A$ as a set of row vectors and $B$ as a set of column vectors, the product matrix $C$ has entries $c_{ij}$ that measure the "alignment" between row $i$ of $A$ and column $j$ of $B$.

Geometrically, multiplying a vector $x$ by a matrix $A$ transforms the vector — stretching, rotating, reflecting, or shearing it. The columns of $A$ are the images of the standard basis vectors. If you visualise the grid lines of 2D space, multiplying by $A$ deforms the grid. The matrix $AB$ combines two such deformations into one.

## Common Mistakes

1. **Multiplying matrices with mismatched inner dimensions**: This is the #1 mistake. Always check that columns of $A$ = rows of $B$. Remember: $(m \times n)(n \times p) = m \times p$.

2. **Assuming commutativity**: $AB$ almost never equals $BA$. Even for square matrices, they differ. Never swap the order without justification.

3. **Confusing matrix multiplication with element-wise multiplication**: The $\cdot$ or juxtaposition typically means matrix multiplication (dot products), not $a_{ij} \cdot b_{ij}$. In NumPy, use `@` for matrix multiplication and `*` for element-wise.

4. **Forgetting that matrix multiplication is not defined for all pairs**: Unlike addition, even matrices of the same size may not be multiplicatively compatible if the inner dimensions don't match.

5. **Misapplying the associative property with transposes**: $(AB)^T = B^T A^T$, not $A^T B^T$. The order reverses.

6. **Thinking $AB = 0$ implies $A = 0$ or $B = 0$**: Unlike with scalars, the product of two non-zero matrices can be the zero matrix.

7. **Confusing row and column order in the dot product**: When computing $c_{ij}$, it is always row $i$ of $A$ times column $j$ of $B$, never column of $A$ times row of $B$.

## Interview Questions

### Beginner

1. What condition must the dimensions of two matrices satisfy for multiplication to be defined?
2. Compute $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$.
3. Is matrix multiplication commutative? Give a brief explanation.
4. What is the identity matrix and what happens when you multiply any matrix by it?
5. If $A$ is $2 \times 3$ and $B$ is $3 \times 5$, what is the size of $AB$? Can $BA$ be computed?

### Intermediate

1. Compute $\begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \end{bmatrix} \begin{bmatrix} 0.5 & -1 \\ 0.2 & 0.3 \\ -0.5 & 0.7 \end{bmatrix}$.
2. Show that $(AB)^T = B^T A^T$ with a concrete $2 \times 2$ example.
3. Explain how matrix multiplication is used in the forward pass of a fully-connected neural network.
4. Why is dimension compatibility required for matrix multiplication? Explain in terms of the dot product.
5. If $A$ and $B$ are both $n \times n$ matrices and $AB = 0$, does it follow that $A = 0$ or $B = 0$? Justify with an example.

### Advanced

1. Prove that matrix multiplication is associative: $(AB)C = A(BC)$ for matrices of compatible dimensions using summation notation.
2. Derive how convolution in a CNN can be expressed as matrix multiplication (the im2col approach). Why is this useful?
3. In the transformer attention mechanism, the operation $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ involves two matrix multiplications. Explain the role of each multiplication and the dimension requirements.

## Practice Problems

### Easy - 5 Questions

1. Compute $\begin{bmatrix} 1 & 3 \\ 2 & 0 \end{bmatrix} \begin{bmatrix} 4 \\ -1 \end{bmatrix}$.
2. Compute $\begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 1 & 4 \\ -2 & 5 \end{bmatrix}$.
3. Multiply $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 5 & -2 \\ 3 & 7 \end{bmatrix}$.
4. If $A$ is $3 \times 2$ and $B$ is $2 \times 3$, what are the dimensions of $AB$ and $BA$?
5. Compute $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$.

### Medium - 5 Questions

6. Compute $\begin{bmatrix} 2 & 1 & 0 \\ -1 & 3 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 2 & -1 \\ 3 & 4 \end{bmatrix}$.
7. Let $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$. Compute $AB$ and $BA$ and verify they are different.
8. A neural network layer has $W = \begin{bmatrix} 0.2 & -0.1 & 0.3 \\ 0.5 & 0.0 & -0.4 \\ -0.2 & 0.6 & 0.1 \end{bmatrix}$ and input $x = \begin{bmatrix} 1 \\ -2 \\ 0 \end{bmatrix}$. Compute $Wx$.
9. Show that $A(B + C) = AB + AC$ for $A = \begin{bmatrix} 2 & 1 \\ 0 & -1 \end{bmatrix}$, $B = \begin{bmatrix} 1 & 0 \\ 2 & 3 \end{bmatrix}$, $C = \begin{bmatrix} -1 & 4 \\ 0 & 2 \end{bmatrix}$.
10. Given $A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}$ (a diagonal matrix) and $x = \begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix}$, compute $Ax$ and interpret the result.

### Hard - 3 Questions

11. Find two non-zero $2 \times 2$ matrices $A$ and $B$ such that $AB = 0$ but $A \neq 0$ and $B \neq 0$. (These are called zero divisors.)
12. In a transformer, $Q$ is $4 \times 8$, $K$ is $4 \times 8$, and $V$ is $4 \times 8$ (for 4 tokens, 8-dimensional embeddings). What are the dimensions of $QK^T$ and $(QK^T)V$? Compute these products for $Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}$, $K = \begin{bmatrix} 1 & 1 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$, $V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix}$ (simplified to 4 tokens, 2-dimensional embeddings for tractability).
13. Prove that $(AB)^T = B^T A^T$ using summation notation for general $m \times n$ and $n \times p$ matrices.

## Solutions

### Easy Solutions

1. $\begin{bmatrix} 1(4) + 3(-1) \\ 2(4) + 0(-1) \end{bmatrix} = \begin{bmatrix} 4 - 3 \\ 8 + 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 8 \end{bmatrix}$

2. $\begin{bmatrix} 2(1)+(-1)(-2) & 2(4)+(-1)(5) \\ 0(1)+3(-2) & 0(4)+3(5) \end{bmatrix} = \begin{bmatrix} 2+2 & 8-5 \\ 0-6 & 0+15 \end{bmatrix} = \begin{bmatrix} 4 & 3 \\ -6 & 15 \end{bmatrix}$

3. $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 5 & -2 \\ 3 & 7 \end{bmatrix} = \begin{bmatrix} 5 & -2 \\ 3 & 7 \end{bmatrix}$ (identity matrix leaves it unchanged)

4. $AB$: $3 \times 3$ (since $3 \times 2$ times $2 \times 3$). $BA$: $2 \times 2$ (since $2 \times 3$ times $3 \times 2$).

5. $[1(4) + 2(5) + 3(6)] = [4 + 10 + 18] = [32]$ (a $1 \times 1$ matrix, i.e., a scalar)

### Medium Solutions

6. $A$ is $2 \times 3$, $B$ is $3 \times 2$, result is $2 \times 2$.

   Row 1: $[(2)(1)+(1)(2)+(0)(3),\; (2)(0)+(1)(-1)+(0)(4)] = [2+2+0,\; 0-1+0] = [4, -1]$
   Row 2: $[(-1)(1)+(3)(2)+(2)(3),\; (-1)(0)+(3)(-1)+(2)(4)] = [-1+6+6,\; 0-3+8] = [11, 5]$

   Result: $\begin{bmatrix} 4 & -1 \\ 11 & 5 \end{bmatrix}$

7. $AB = \begin{bmatrix} 1(5)+2(7) & 1(6)+2(8) \\ 3(5)+4(7) & 3(6)+4(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$

   $BA = \begin{bmatrix} 5(1)+6(3) & 5(2)+6(4) \\ 7(1)+8(3) & 7(2)+8(4) \end{bmatrix} = \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix}$

   They are different.

8. $Wx = \begin{bmatrix} 0.2(1)+(-0.1)(-2)+0.3(0) \\ 0.5(1)+0.0(-2)+(-0.4)(0) \\ (-0.2)(1)+0.6(-2)+0.1(0) \end{bmatrix} = \begin{bmatrix} 0.2+0.2+0 \\ 0.5+0+0 \\ -0.2-1.2+0 \end{bmatrix} = \begin{bmatrix} 0.4 \\ 0.5 \\ -1.4 \end{bmatrix}$

9. $B+C = \begin{bmatrix} 0 & 4 \\ 2 & 5 \end{bmatrix}$. $A(B+C) = \begin{bmatrix} 2(0)+1(2) & 2(4)+1(5) \\ 0(0)+(-1)(2) & 0(4)+(-1)(5) \end{bmatrix} = \begin{bmatrix} 2 & 13 \\ -2 & -5 \end{bmatrix}$.

   $AB = \begin{bmatrix} 2(1)+1(2) & 2(0)+1(3) \\ 0(1)+(-1)(2) & 0(0)+(-1)(3) \end{bmatrix} = \begin{bmatrix} 4 & 3 \\ -2 & -3 \end{bmatrix}$

   $AC = \begin{bmatrix} 2(-1)+1(0) & 2(4)+1(2) \\ 0(-1)+(-1)(0) & 0(4)+(-1)(2) \end{bmatrix} = \begin{bmatrix} -2 & 10 \\ 0 & -2 \end{bmatrix}$

   $AB + AC = \begin{bmatrix} 4-2 & 3+10 \\ -2+0 & -3-2 \end{bmatrix} = \begin{bmatrix} 2 & 13 \\ -2 & -5 \end{bmatrix}$. Matches $A(B+C)$.

10. $Ax = \begin{bmatrix} 1(2) \\ 2(3) \\ 3(4) \end{bmatrix} = \begin{bmatrix} 2 \\ 6 \\ 12 \end{bmatrix}$. The diagonal matrix scales each component of $x$ by the corresponding diagonal entry. This is equivalent to the element-wise product of the diagonal with the vector.

### Hard Solutions

11. One example: $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$, $B = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$. Then $AB = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$ but neither $A$ nor $B$ is zero. Another example: $A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$, $B = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$. Then $AB = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$.

12. $Q$ is $4 \times 2$, $K$ is $4 \times 2$ (simplified embeddings). For $QK^T$: $K^T$ is $2 \times 4$, so $QK^T$ is $4 \times 4$. $(QK^T)V$: $QK^T$ is $4 \times 4$, $V$ is $4 \times 2$, so the result is $4 \times 2$.

    First, $K^T = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix}$.

    $QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix}
    1(1)+0(1) & 1(0)+0(0) & 1(1)+0(0) & 1(0)+0(1) \\
    0(1)+1(1) & 0(0)+1(0) & 0(1)+1(0) & 0(0)+1(1) \\
    1(1)+1(1) & 1(0)+1(0) & 1(1)+1(0) & 1(0)+1(1) \\
    0(1)+0(1) & 0(0)+0(0) & 0(1)+0(0) & 0(0)+0(1)
    \end{bmatrix} = \begin{bmatrix}
    1 & 0 & 1 & 0 \\
    1 & 0 & 0 & 1 \\
    2 & 0 & 1 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}$

    Then $(QK^T)V = \begin{bmatrix}
    1 & 0 & 1 & 0 \\
    1 & 0 & 0 & 1 \\
    2 & 0 & 1 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix}
    1(1)+0(0)+1(0)+0(1) & 1(0)+0(1)+1(0)+0(1) \\
    1(1)+0(0)+0(0)+1(1) & 1(0)+0(1)+0(0)+1(1) \\
    2(1)+0(0)+1(0)+1(1) & 2(0)+0(1)+1(0)+1(1) \\
    0(1)+0(0)+0(0)+0(1) & 0(0)+0(1)+0(0)+0(1)
    \end{bmatrix} = \begin{bmatrix}
    1 & 0 \\
    2 & 1 \\
    3 & 1 \\
    0 & 0
    \end{bmatrix}$

13. Let $A$ be $m \times n$ and $B$ be $n \times p$. Let $C = AB$, so $c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$. Then $C^T$ has entries $(C^T)_{ji} = c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$.

    Now consider $B^T A^T$. $B^T$ is $p \times n$ with entries $(B^T)_{jk} = b_{kj}$. $A^T$ is $n \times m$ with entries $(A^T)_{ki} = a_{ik}$. The product $B^T A^T$ is $p \times m$, and its entry at position $(j, i)$ is:

    $$
    (B^T A^T)_{ji} = \sum_{k=1}^{n} (B^T)_{jk} (A^T)_{ki} = \sum_{k=1}^{n} b_{kj} a_{ik} = \sum_{k=1}^{n} a_{ik} b_{kj} = (C^T)_{ji}
    $$

    This holds for all $i, j$, therefore $(AB)^T = B^T A^T$.

## Related Concepts

- **Dot Product** (MATH-016): The fundamental operation used in each entry of matrix multiplication
- **Matrix Addition** (MATH-021): Used alongside multiplication in $Wx + b$
- **Scalar Multiplication**: Multiplying a matrix by a constant
- **Identity Matrix**: The multiplicative identity for matrix multiplication
- **Matrix Inverse**: The matrix $A^{-1}$ such that $A^{-1}A = I$
- **Linear Transformations**: Matrix multiplication encodes composition of linear maps
- **Transpose** (MATH-024): $(AB)^T = B^T A^T$

## Next Concepts

- **MATH-024 Transpose of a Matrix**: Often used in conjunction with multiplication in ML formulas
- **MATH-025 Determinant**: A scalar associated with square matrices
- **MATH-026 Matrix Inverse**: Solving $Ax = b$ and understanding when it exists
- **MATH-027 Eigenvalues and Eigenvectors**: The "characteristic" vectors of a matrix that are only scaled, not rotated
- **MATH-028 Singular Value Decomposition (SVD)** : Factoring any matrix into three simpler matrices

## Summary

Matrix multiplication is a non-commutative, associative operation that computes each entry of the product $AB$ as the dot product of a row of $A$ and a column of $B$. The inner dimensions must match ($n$ in $m \times n$ and $n \times p$), and the result has the outer dimensions ($m \times p$). Matrix multiplication is not element-wise — it is fundamentally different from scalar multiplication. It encodes the composition of linear transformations and is the computational core of neural networks, attention mechanisms, and virtually every algorithm in machine learning and scientific computing.

## Key Takeaways

- $(AB)_{ij} = \sum_{k} a_{ik} b_{kj}$ — each entry is a dot product of a row and a column
- Dimension rule: $(m \times n)(n \times p) = m \times p$ — inner dimensions must match
- Matrix multiplication is **not commutative**: $AB \neq BA$ in general
- It is associative: $(AB)C = A(BC)$
- The identity matrix $I$ satisfies $AI = IA = A$
- Matrix multiplication distributes over addition: $A(B+C) = AB + AC$
- In AI/ML: forward pass ($y = Wx$), attention ($QK^T$), convolutions (im2col), and PCA ($X^T X$) all rely on matrix multiplication
- Always verify dimension compatibility before multiplying
