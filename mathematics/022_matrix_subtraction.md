# Concept: Matrix Subtraction

## Concept ID

MATH-022

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define matrix subtraction as element-wise subtraction of corresponding entries
- Relate matrix subtraction to addition with the additive inverse
- Verify dimension compatibility before subtracting matrices
- Interpret matrix subtraction geometrically as vector difference
- Apply matrix subtraction to compute error, residuals, and gradient updates in ML

## Prerequisites

- Understanding of matrix addition (MATH-021)
- Familiarity with negative numbers and integer subtraction
- Basic matrix notation: $a_{ij}$ represents the entry in row $i$, column $j$

## Definition

**Matrix subtraction** is the operation of subtracting one matrix from another by subtracting their corresponding entries. If $A$ and $B$ are both $m \times n$ matrices, their difference $D = A - B$ is also an $m \times n$ matrix where:

$$
d_{ij} = a_{ij} - b_{ij}
$$

for $i = 1, 2, \dots, m$ and $j = 1, 2, \dots, n$.

As with addition, both matrices must have the same dimensions. If $A$ and $B$ are not the same size, $A - B$ is undefined.

Matrix subtraction can be understood as addition with the additive inverse:

$$
A - B = A + (-B)
$$

where $-B$ is the matrix whose entries are $-b_{ij}$.

For example, if:

$$
A = \begin{bmatrix} 8 & 3 \\ 2 & -5 \end{bmatrix}, \quad
B = \begin{bmatrix} 5 & 1 \\ 4 & -2 \end{bmatrix}
$$

then:

$$
A - B = \begin{bmatrix} 8-5 & 3-1 \\ 2-4 & -5-(-2) \end{bmatrix}
= \begin{bmatrix} 3 & 2 \\ -2 & -3 \end{bmatrix}
$$

## Intuition

Matrix subtraction tells you how two matrices differ entry by entry. If matrix $A$ represents actual sales and matrix $B$ represents predicted sales, then $A - B$ gives the error for each product-region combination — positive where actual exceeds prediction, negative where prediction exceeds actual.

Just like subtracting two numbers tells you the distance and direction between them on a number line, subtracting two matrices tells you the difference between two datasets element by element.

## Why This Concept Matters

Matrix subtraction is fundamental for:

- **Error and residual computation**: $y - \hat{y}$ (actual minus predicted) is a matrix subtraction in multi-output regression and classification
- **Gradient updates**: Gradient descent updates weights as $W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla W$, which involves matrix subtraction
- **Difference analysis**: Comparing two datasets, time points, or model states
- **Change detection**: Highlighting what changed between two images or signals
- **Numerical optimisation**: Computing updates, corrections, and adjustments

## Historical Background

Matrix subtraction was implicitly defined alongside matrix addition by **Arthur Cayley** in his 1858 "A Memoir on the Theory of Matrices." Since subtraction is simply addition of the negative, the historical development follows that of matrix addition. As matrices gained prominence in physics and engineering during the late 19th and early 20th centuries, matrix subtraction became essential for solving systems of equations, computing errors in least-squares problems, and later, for training machine learning models via gradient descent.

## Real World Examples

1. **Sales vs Forecast**: A retail company compares actual monthly sales (matrix $A$) against forecasted sales (matrix $B$). The difference $A - B$ highlights which products and regions exceeded or fell short of expectations.

2. **Medical Imaging**: A radiologist subtracts a pre-contrast image matrix from a post-contrast image matrix to highlight areas where contrast agent accumulated (e.g., tumours).

3. **Change Detection**: Satellite images of the same location taken on different dates are subtracted to reveal changes — new buildings, deforestation, flood damage.

4. **Budget Variance**: A company's budgeted expenses matrix subtracted from actual expenses shows where overspending or underspending occurred.

5. **Quality Control**: A manufacturer subtracts a reference product's specification matrix from each unit's measurement matrix to detect deviations from the standard.

## AI/ML Relevance

Matrix subtraction is a core operation in nearly every machine learning algorithm.

1. **Computing Residuals/Errors**: In regression tasks, the error is $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$, where $\mathbf{y}$ is the matrix of true labels and $\hat{\mathbf{y}}$ is the matrix of predictions. For a batch of $m$ samples with $k$ outputs, both are $m \times k$ matrices. The residual matrix tells us how far off each prediction is for each output dimension.

2. **Gradient Descent Updates**: The core update rule for training neural networks is:
   $$
   W^{(t+1)} = W^{(t)} - \eta \cdot \nabla_W \mathcal{L}
   $$
   Here, $W^{(t)}$ is the current weight matrix, $\eta$ is the learning rate, and $\nabla_W \mathcal{L}$ is the gradient matrix. The subtraction moves the weights in the direction that reduces loss. Every epoch involves many such matrix subtractions.

3. **Loss Function Computation**: Mean squared error (MSE) for multi-output problems:
   $$
   \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \| \mathbf{y}_i - \hat{\mathbf{y}}_i \|^2
   $$
   The inner difference $\mathbf{y}_i - \hat{\mathbf{y}}_i$ is a vector subtraction (or matrix subtraction for a batch).

4. **Contrastive Learning**: In Siamese networks, the difference between embedding matrices of positive and negative pairs is computed to formulate contrastive loss: $\mathcal{L} = \frac{1}{2} \| E_1 - E_2 \|^2$.

5. **Adversarial Attacks**: Generating adversarial examples involves adding a small perturbation matrix $\delta$ to the input: $x_{\text{adv}} = x + \delta$. The perturbation itself is often computed by subtracting the original input from a target representation.

6. **Batch Normalisation**: The normalisation step subtracts the batch mean matrix from the input: $\hat{x} = x - \mu_{\text{batch}}$.

## Mathematical Explanation

Matrix subtraction is defined element-wise for matrices of the same dimensions. Given $A$ and $B$, both $m \times n$:

$$
D = A - B \quad \text{where} \quad d_{ij} = a_{ij} - b_{ij}
$$

This is equivalent to $A + (-B)$, since $a_{ij} - b_{ij} = a_{ij} + (-b_{ij})$. Therefore, every property that follows from matrix addition also applies to subtraction through this relationship.

### Relationship to Matrix Addition

While addition is commutative ($A + B = B + A$), subtraction is **not** commutative:

$$
A - B \neq B - A \quad \text{(in general)}
$$

Instead, $A - B = -(B - A)$. Swapping the order negates the result.

If $A = B$, then $A - A = 0$, the zero matrix.

### Geometric Interpretation

If each column of a matrix represents a point in $n$-dimensional space, then subtracting two matrices compares their corresponding columns. For a single column view, if $\mathbf{a}$ and $\mathbf{b}$ are column vectors of the same dimension, $\mathbf{a} - \mathbf{b}$ is the vector from the tip of $\mathbf{b}$ to the tip of $\mathbf{a}$ (when both are placed with tails at the origin). The length of this difference vector is the distance between the two points.

When extended to matrices, $A - B$ gives, column by column, the displacement vectors between corresponding points in two datasets.

## Formula(s)

**General formula for matrix subtraction** (both matrices $m \times n$):

$$
(A - B)_{ij} = a_{ij} - b_{ij}
$$

**Expressed as addition of the negative:**

$$
A - B = A + (-B)
$$

**Writing out a $3 \times 2$ case:**

$$
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}
-
\begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \\ b_{31} & b_{32} \end{bmatrix}
=
\begin{bmatrix} a_{11}-b_{11} & a_{12}-b_{12} \\ a_{21}-b_{21} & a_{22}-b_{22} \\ a_{31}-b_{31} & a_{32}-b_{32} \end{bmatrix}
$$

## Properties

1. **Closure**: If $A$ and $B$ are $m \times n$ matrices, then $A - B$ is also $m \times n$.

2. **Not Commutative**: $A - B \neq B - A$ in general. Instead, $A - B = -(B - A)$.

3. **Not Associative**: $(A - B) - C \neq A - (B - C)$ in general. Compare: $(A - B) - C = A - B - C$, but $A - (B - C) = A - B + C$.

4. **Relation to Addition**: $A - B = A + (-B)$, where $-B$ is the additive inverse of $B$.

5. **Self-Subtraction**: $A - A = 0$ (the zero matrix).

6. **Distributive Property**: $c(A - B) = cA - cB$ for any scalar $c$.

7. **Compatibility with Transpose**: $(A - B)^T = A^T - B^T$.

## Step-by-Step Worked Examples

### Example 1: Basic Matrix Subtraction ($2 \times 2$)

**Problem:** Let

$$
A = \begin{bmatrix} 10 & 4 \\ 7 & -2 \end{bmatrix}, \quad
B = \begin{bmatrix} 3 & 8 \\ 5 & -6 \end{bmatrix}
$$

Compute $A - B$.

**Solution:**

**Step 1:** Check dimensions. Both are $2 \times 2$, so subtraction is defined.

**Step 2:** Subtract corresponding entries:

- $d_{11} = 10 - 3 = 7$
- $d_{12} = 4 - 8 = -4$
- $d_{21} = 7 - 5 = 2$
- $d_{22} = -2 - (-6) = -2 + 6 = 4$

**Step 3:** Assemble:

$$
A - B = \begin{bmatrix} 7 & -4 \\ 2 & 4 \end{bmatrix}
$$

**Step 4:** Verify using $A + (-B)$. First, $-B = \begin{bmatrix} -3 & -8 \\ -5 & 6 \end{bmatrix}$. Then:

$$
A + (-B) = \begin{bmatrix} 10-3 & 4-8 \\ 7-5 & -2+6 \end{bmatrix}
= \begin{bmatrix} 7 & -4 \\ 2 & 4 \end{bmatrix}
$$

This matches.

### Example 2: Computing the Difference Between Actual and Predicted Values

**Problem:** A model makes predictions for 2 samples with 3 outputs each. The true values and predictions are:

$$
Y_{\text{true}} = \begin{bmatrix} 0.9 & 0.1 & 0.0 \\ 0.2 & 0.7 & 0.1 \end{bmatrix}, \quad
Y_{\text{pred}} = \begin{bmatrix} 0.8 & 0.2 & 0.0 \\ 0.3 & 0.5 & 0.2 \end{bmatrix}
$$

Compute the error matrix $E = Y_{\text{true}} - Y_{\text{pred}}$.

**Solution:**

**Step 1:** Both matrices are $2 \times 3$. Subtraction is defined.

**Step 2:** Subtract entry by entry:

Row 1: $[0.9-0.8,\; 0.1-0.2,\; 0.0-0.0] = [0.1, -0.1, 0.0]$
Row 2: $[0.2-0.3,\; 0.7-0.5,\; 0.1-0.2] = [-0.1, 0.2, -0.1]$

**Step 3:** Assemble:

$$
E = \begin{bmatrix} 0.1 & -0.1 & 0.0 \\ -0.1 & 0.2 & -0.1 \end{bmatrix}
$$

**Step 4:** Interpretation: For sample 1, output 1 was underestimated (error $+0.1$), output 2 was overestimated (error $-0.1$), and output 3 was exact. For sample 2, output 1 was overestimated, output 2 underestimated, and output 3 overestimated.

### Example 3: Gradient Descent Update

**Problem:** A weight matrix $W$ at iteration $t$ is:

$$
W^{(t)} = \begin{bmatrix} 0.5 & -0.2 \\ 0.3 & 0.8 \\ -0.1 & 0.4 \end{bmatrix}
$$

The gradient of the loss with respect to $W$ is:

$$
\nabla_W \mathcal{L} = \begin{bmatrix} 0.02 & -0.01 \\ -0.03 & 0.02 \\ 0.01 & 0.03 \end{bmatrix}
$$

The learning rate is $\eta = 0.1$. Compute the updated weight matrix $W^{(t+1)} = W^{(t)} - \eta \nabla_W \mathcal{L}$.

**Solution:**

**Step 1:** Compute $\eta \nabla_W \mathcal{L}$:

$$
\eta \nabla_W \mathcal{L} = 0.1 \times \begin{bmatrix} 0.02 & -0.01 \\ -0.03 & 0.02 \\ 0.01 & 0.03 \end{bmatrix}
= \begin{bmatrix} 0.002 & -0.001 \\ -0.003 & 0.002 \\ 0.001 & 0.003 \end{bmatrix}
$$

**Step 2:** Subtract from $W^{(t)}$:

Row 1: $[0.5-0.002,\; -0.2-(-0.001)] = [0.498, -0.199]$
Row 2: $[0.3-(-0.003),\; 0.8-0.002] = [0.303, 0.798]$
Row 3: $[-0.1-0.001,\; 0.4-0.003] = [-0.101, 0.397]$

**Step 3:** Assemble:

$$
W^{(t+1)} = \begin{bmatrix} 0.498 & -0.199 \\ 0.303 & 0.798 \\ -0.101 & 0.397 \end{bmatrix}
$$

**Step 4:** Notice how the weights have been slightly adjusted in the direction that reduces the loss. After many such updates, the model converges.

### Example 4: Non-Commutativity of Subtraction

**Problem:** Let $A = \begin{bmatrix} 1 & 4 \\ 2 & 5 \end{bmatrix}$, $B = \begin{bmatrix} 3 & 0 \\ -1 & 2 \end{bmatrix}$. Show that $A - B \neq B - A$.

**Solution:**

**Step 1:** Compute $A - B$:

$$
A - B = \begin{bmatrix} 1-3 & 4-0 \\ 2-(-1) & 5-2 \end{bmatrix}
= \begin{bmatrix} -2 & 4 \\ 3 & 3 \end{bmatrix}
$$

**Step 2:** Compute $B - A$:

$$
B - A = \begin{bmatrix} 3-1 & 0-4 \\ -1-2 & 2-5 \end{bmatrix}
= \begin{bmatrix} 2 & -4 \\ -3 & -3 \end{bmatrix}
$$

**Step 3:** Compare. $A - B = \begin{bmatrix} -2 & 4 \\ 3 & 3 \end{bmatrix}$ and $B - A = \begin{bmatrix} 2 & -4 \\ -3 & -3 \end{bmatrix}$. They are negatives of each other: $B - A = -(A - B)$.

### Example 5: Subtracting to Find the Additive Inverse

**Problem:** If $A + X = B$, where $A = \begin{bmatrix} 2 & -1 \\ 3 & 0 \\ -4 & 5 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 2 \\ 1 & 4 \\ -2 & 1 \end{bmatrix}$, find $X$.

**Solution:**

**Step 1:** Solve the matrix equation $A + X = B$ for $X$. Subtract $A$ from both sides:

$$
X = B - A
$$

**Step 2:** Compute $B - A$:

Row 1: $[5-2,\; 2-(-1)] = [3, 3]$
Row 2: $[1-3,\; 4-0] = [-2, 4]$
Row 3: $[-2-(-4),\; 1-5] = [2, -4]$

**Step 3:**

$$
X = \begin{bmatrix} 3 & 3 \\ -2 & 4 \\ 2 & -4 \end{bmatrix}
$$

**Step 4:** Verify: $A + X = \begin{bmatrix} 2+3 & -1+3 \\ 3+(-2) & 0+4 \\ -4+2 & 5+(-4) \end{bmatrix} = \begin{bmatrix} 5 & 2 \\ 1 & 4 \\ -2 & 1 \end{bmatrix} = B$. Correct.

## Visual Interpretation

Geometrically, if each column of a matrix is a point in space, $A - B$ gives the vectors that point from each point in $B$ to the corresponding point in $A$. For $2 \times 2$ matrices, plot the two columns of $A$ and $B$ as points in the plane. The arrows from $B$'s points to $A$'s points are the column vectors of $A - B$.

For image data, subtracting one image matrix from another highlights differences. If you subtract a background image from a current frame in video surveillance, the non-zero entries show where motion occurred. The magnitude of the difference indicates how much the pixel value changed.

For datasets, think of $A - B$ as a "difference map" — positive entries mean $A$ has larger values, negative entries mean $B$ has larger values, and zeros mean they are identical at that position.

## Common Mistakes

1. **Subtracting matrices of different sizes**: As with addition, $A - B$ is only defined when $A$ and $B$ have the same dimensions. Always verify sizes first.

2. **Assuming commutativity**: $A - B \neq B - A$ in general. The order matters just as with scalar subtraction.

3. **Misapplying associativity**: $(A - B) - C \neq A - (B - C)$ because $A - (B - C) = A - B + C$. Use parentheses carefully.

4. **Forgetting to distribute the negative sign**: $A - (B + C) = A - B - C$, not $A - B + C$. The subtraction distributes as $-$ across all terms.

5. **Confusing $A - B$ with $B - A$**: They are negatives of each other. If you compute the error as $y - \hat{y}$, you must keep the order consistent.

6. **Thinking subtraction is just like addition**: While element-wise, subtraction requires careful handling of signs. $a_{ij} - b_{ij}$ can be positive, negative, or zero.

7. **Neglecting sign when computing gradients**: In $W - \eta \nabla W$, the subtraction moves weights opposite the gradient direction. Using addition would increase the loss instead of decreasing it.

## Interview Questions

### Beginner

1. How is matrix subtraction defined? What condition must the matrices satisfy?
2. Compute $\begin{bmatrix} 7 & 1 \\ -2 & 4 \end{bmatrix} - \begin{bmatrix} 3 & 5 \\ 6 & -1 \end{bmatrix}$.
3. Is matrix subtraction commutative? Explain with an example.
4. What is $A - A$ for any matrix $A$?
5. If $A$ is $3 \times 2$ and $B$ is $2 \times 3$, can you compute $A - B$? Explain.

### Intermediate

1. Show that $A - B = A + (-B)$ using the definition of matrix subtraction.
2. Given $A = \begin{bmatrix} 4 & -1 \\ 2 & 0 \\ 3 & 5 \end{bmatrix}$ and $A - B = \begin{bmatrix} 1 & 2 \\ -1 & 3 \\ 0 & 1 \end{bmatrix}$, find $B$.
3. How is matrix subtraction used in the gradient descent update rule for neural network weights?
4. Prove that $(A - B)^T = A^T - B^T$ for $2 \times 2$ matrices.
5. If $X = A - B$ and $Y = B - A$, what is the relationship between $X$ and $Y$?

### Advanced

1. Derive the relationship between matrix subtraction and the mean squared error loss in multi-output regression.
2. In batch normalisation, the input $x$ is normalised as $\hat{x} = \frac{x - \mu}{\sigma}$. Explain the role of matrix/vector subtraction in this formula and why it is essential for training stability.
3. Prove that the set of $m \times n$ matrices under subtraction does NOT form a group (hint: consider closure and the need for an identity under subtraction specifically).

## Practice Problems

### Easy - 5 Questions

1. Compute $\begin{bmatrix} 8 & 3 \\ 1 & -5 \end{bmatrix} - \begin{bmatrix} 2 & 7 \\ -4 & 2 \end{bmatrix}$.
2. Compute $\begin{bmatrix} 5 & -1 & 0 \\ 2 & 3 & -4 \end{bmatrix} - \begin{bmatrix} 1 & 2 & 3 \\ -1 & 0 & 2 \end{bmatrix}$.
3. Find $B$ such that $A - B = 0$ where $A = \begin{bmatrix} 3 & -2 \\ 1 & 4 \end{bmatrix}$.
4. Compute $-\left( \begin{bmatrix} 4 & -3 \\ 2 & 1 \end{bmatrix} - \begin{bmatrix} 1 & 5 \\ -2 & 3 \end{bmatrix} \right)$.
5. If $A = \begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 15 \\ 25 & 35 \end{bmatrix}$, compute $A - B$ and $B - A$. What do you notice?

### Medium - 5 Questions

6. Solve for $X$: $\begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix} - X = \begin{bmatrix} -1 & 2 \\ 4 & 1 \end{bmatrix}$.
7. Given $A = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.3 \end{bmatrix}$ and gradient $\nabla W = \begin{bmatrix} 0.01 & 0.02 \\ -0.01 & 0.03 \end{bmatrix}$ with $\eta = 0.5$, compute $W_{\text{new}} = A - \eta \nabla W$.
8. Show that $(A - B) - C \neq A - (B - C)$ using $2 \times 2$ matrices of your choice.
9. True labels: $Y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$, predictions: $\hat{Y} = \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.7 \\ 0.9 & 0.8 \end{bmatrix}$. Compute the residual matrix $E = Y - \hat{Y}$.
10. Verify that $c(A - B) = cA - cB$ for $c = 3$, $A = \begin{bmatrix} 2 & 0 \\ -1 & 4 \end{bmatrix}$, $B = \begin{bmatrix} 1 & 2 \\ 3 & -2 \end{bmatrix}$.

### Hard - 3 Questions

11. Prove that $(A - B)^T = A^T - B^T$ for general $m \times n$ matrices using index notation.
12. In a linear regression model with 3 targets, the weight matrix $W$ is $5 \times 3$. After one gradient step, $W_{\text{new}} = W - \eta \nabla W$. If $W$ and $\nabla W$ are given, derive an expression for the change in the loss to first order (use the concept of matrix inner product).
13. A $3 \times 3$ matrix $A$ is subtracted from the identity matrix: $I - A$. If $A$ has entries $a_{ij} = 0.2$ for all $i, j$, compute $I - A$ and interpret what this matrix represents in terms of a "dampening" or "leakage" transformation.

## Solutions

### Easy Solutions

1. $\begin{bmatrix} 8-2 & 3-7 \\ 1-(-4) & -5-2 \end{bmatrix} = \begin{bmatrix} 6 & -4 \\ 5 & -7 \end{bmatrix}$

2. $\begin{bmatrix} 5-1 & -1-2 & 0-3 \\ 2-(-1) & 3-0 & -4-2 \end{bmatrix} = \begin{bmatrix} 4 & -3 & -3 \\ 3 & 3 & -6 \end{bmatrix}$

3. $B = A = \begin{bmatrix} 3 & -2 \\ 1 & 4 \end{bmatrix}$ (since $A - B = 0 \implies B = A$)

4. First compute inner: $\begin{bmatrix} 4-1 & -3-5 \\ 2-(-2) & 1-3 \end{bmatrix} = \begin{bmatrix} 3 & -8 \\ 4 & -2 \end{bmatrix}$. Then negate: $\begin{bmatrix} -3 & 8 \\ -4 & 2 \end{bmatrix}$

5. $A - B = \begin{bmatrix} 5 & 5 \\ 5 & 5 \end{bmatrix}$, $B - A = \begin{bmatrix} -5 & -5 \\ -5 & -5 \end{bmatrix}$. They are negatives of each other.

### Medium Solutions

6. $X = \begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix} - \begin{bmatrix} -1 & 2 \\ 4 & 1 \end{bmatrix} = \begin{bmatrix} 3 & -3 \\ -4 & 2 \end{bmatrix}$

7. $\eta \nabla W = 0.5 \times \begin{bmatrix} 0.01 & 0.02 \\ -0.01 & 0.03 \end{bmatrix} = \begin{bmatrix} 0.005 & 0.01 \\ -0.005 & 0.015 \end{bmatrix}$. Then $W_{\text{new}} = \begin{bmatrix} 0.5-0.005 & -0.2-0.01 \\ 0.1-(-0.005) & 0.3-0.015 \end{bmatrix} = \begin{bmatrix} 0.495 & -0.21 \\ 0.105 & 0.285 \end{bmatrix}$.

8. Use $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $B = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$, $C = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$. $(A-B)-C = \begin{bmatrix} -1 & -1 \\ -1 & -1 \end{bmatrix} - \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} -2 & -2 \\ -2 & -2 \end{bmatrix}$. $A-(B-C) = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} - \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$. Not equal.

9. $E = \begin{bmatrix} 1-0.8 & 0-0.2 \\ 0-0.3 & 1-0.7 \\ 1-0.9 & 1-0.8 \end{bmatrix} = \begin{bmatrix} 0.2 & -0.2 \\ -0.3 & 0.3 \\ 0.1 & 0.2 \end{bmatrix}$

10. $A - B = \begin{bmatrix} 1 & -2 \\ -4 & 6 \end{bmatrix}$, $c(A-B) = 3 \times \begin{bmatrix} 1 & -2 \\ -4 & 6 \end{bmatrix} = \begin{bmatrix} 3 & -6 \\ -12 & 18 \end{bmatrix}$. $cA = \begin{bmatrix} 6 & 0 \\ -3 & 12 \end{bmatrix}$, $cB = \begin{bmatrix} 3 & 6 \\ 9 & -6 \end{bmatrix}$, $cA - cB = \begin{bmatrix} 3 & -6 \\ -12 & 18 \end{bmatrix}$. They match.

### Hard Solutions

11. Proof using index notation. Let $D = A - B$. Then $d_{ij} = a_{ij} - b_{ij}$. The transpose $D^T$ has entries $(D^T)_{ji} = d_{ij} = a_{ij} - b_{ij}$. Meanwhile, $A^T$ has entries $(A^T)_{ji} = a_{ij}$, and $B^T$ has entries $(B^T)_{ji} = b_{ij}$. Therefore $(A^T - B^T)_{ji} = (A^T)_{ji} - (B^T)_{ji} = a_{ij} - b_{ij} = (D^T)_{ji}$. This holds for all $i, j$, so $(A - B)^T = A^T - B^T$.

12. Let $\Delta W = -\eta \nabla W$ be the change. The loss $\mathcal{L}(W)$ changes to first order as $\Delta \mathcal{L} \approx \langle \nabla W, \Delta W \rangle_F$, where $\langle \cdot, \cdot \rangle_F$ is the Frobenius inner product (sum of entry-wise products). Substituting: $\Delta \mathcal{L} \approx \langle \nabla W, -\eta \nabla W \rangle_F = -\eta \|\nabla W\|_F^2 \leq 0$. This shows the loss decreases (to first order) when we subtract the gradient scaled by $\eta$.

13. $A$ is all $0.2$: $A = \begin{bmatrix} 0.2 & 0.2 & 0.2 \\ 0.2 & 0.2 & 0.2 \\ 0.2 & 0.2 & 0.2 \end{bmatrix}$. $I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$. So:
    $$
    I - A = \begin{bmatrix} 0.8 & -0.2 & -0.2 \\ -0.2 & 0.8 & -0.2 \\ -0.2 & -0.2 & 0.8 \end{bmatrix}
    $$
    This matrix represents a transformation where each output is $0.8$ of its own input minus $0.2$ of each other input. In a network, this could represent a lateral inhibition or leakage mechanism, where each neuron suppresses its neighbours while retaining most of its own signal.

## Related Concepts

- **Matrix Addition** (MATH-021): The companion operation; subtraction is addition with the additive inverse
- **Scalar Multiplication**: Multiplying every entry of a matrix by a constant
- **Vector Subtraction** (MATH-012): The 1D analogue of matrix subtraction
- **Gradient Descent**: The optimisation algorithm that relies on matrix subtraction for weight updates
- **Residual Analysis**: Using subtraction to evaluate model performance

## Next Concepts

- **MATH-023 Matrix Multiplication**: The next major matrix operation — not element-wise and more powerful
- **MATH-013 Scalar Multiplication**: Scaling matrices, which combines with subtraction in gradient updates
- **MATH-024 Transpose of a Matrix**: Often applied alongside subtraction in ML formulas

## Summary

Matrix subtraction is the element-wise operation $A - B$ where each entry $d_{ij} = a_{ij} - b_{ij}$, defined only when $A$ and $B$ have identical dimensions. It is equivalent to adding the additive inverse: $A - B = A + (-B)$. Unlike addition, subtraction is not commutative. Matrix subtraction is crucial for computing errors, residuals, and gradient updates in machine learning, and for comparing datasets across time, space, or model predictions.

## Key Takeaways

- $A - B$ requires $A$ and $B$ to have the same dimensions $m \times n$
- $(A - B)_{ij} = a_{ij} - b_{ij}$ (element-wise subtraction)
- $A - B = A + (-B)$, linking subtraction to addition
- Subtraction is NOT commutative: $A - B \neq B - A$; instead $A - B = -(B - A)$
- $A - A = 0$ (the zero matrix)
- In AI/ML, subtraction is used for computing residuals ($y - \hat{y}$), gradient descent updates ($W - \eta \nabla W$), and batch normalisation ($x - \mu$)
- Always check dimensions before subtracting — mismatched sizes make the operation undefined
