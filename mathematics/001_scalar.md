# Concept: Scalar

## Concept ID

MATH-001

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Foundations

## Learning Objectives

- Define a scalar and distinguish it from other mathematical objects.
- Identify scalars in everyday contexts and in technical fields.
- Perform basic arithmetic operations with scalars.
- Understand how scalars are used in AI/ML contexts (learning rates, loss values, regularisation parameters).
- Apply scalar operations in step-by-step calculations with confidence.

## Prerequisites

- Basic arithmetic (addition, subtraction, multiplication, division).
- Familiarity with fractions, decimals, and percentages at the school level.
- No prior knowledge of linear algebra or calculus is required.

## Definition

A **scalar** is a single numerical value. It is the simplest kind of mathematical object. Scalars are used to represent quantities that have **magnitude only** — they carry no direction or orientation.

In contrast to a **vector** (which has both magnitude and direction) or a **matrix** (which is a rectangular array of numbers), a scalar stands alone. Examples of scalars include the number 5, $-3.14$, $0$, $\frac{7}{8}$, and $\sqrt{2}$.

Formally, a scalar is an element of a **field** — a set of numbers equipped with addition and multiplication that behave in familiar ways. For most practical purposes in AI and machine learning, the field in question is the set of real numbers $\mathbb{R}$.

## Intuition

Think of a scalar as a **single measurement** or a **single quantity**. If you ask "What is the temperature outside?" and the answer is $30^\circ$C, that $30$ is a scalar. If you ask "How many apples are in this basket?" and the answer is $12$, that $12$ is a scalar. If you ask "What is the price of this book?" and the answer is $\$24.99$, that $24.99$ is a scalar.

Scalars contrast with **vectors**. If you say "Walk 5 kilometres", the 5 is a scalar (the distance). But if you say "Walk 5 kilometres north-east", the "5 kilometres north-east" is a vector because it includes both a magnitude (5) and a direction. The magnitude part (the 5) is itself a scalar.

A useful mental model: a scalar is to a number what a single atom is to a molecule. It is the irreducible, fundamental unit of numerical information.

## Why This Concept Matters

Scalars are the building blocks of all quantitative disciplines. Every mathematical operation in science, engineering, economics, and artificial intelligence ultimately reduces to operations on scalar numbers.

In AI and machine learning specifically:

- **Loss functions** (e.g., mean squared error, cross-entropy loss) produce a **single scalar value** that measures how well a model is performing.
- **Learning rates** (e.g., $\alpha = 0.001$) are scalars that control how much a model's parameters are updated during training.
- **Regularisation parameters** (e.g., $\lambda = 0.01$) are scalars that control the trade-off between fitting the training data and keeping model weights small.
- **Accuracy, precision, recall, F1-score** — all evaluation metrics are scalars.

Without a solid understanding of scalars and scalar operations, it is impossible to understand vectors, matrices, tensors, gradients, or any of the mathematical machinery that powers modern AI.

## Real World Examples

**Example 1: Temperature.** The temperature of a room is $22.5^\circ$C. This single number is a scalar. It tells you how hot or cold the room is, but it does not tell you which direction the heat is flowing.

**Example 2: Speed vs. Velocity.** A car travelling at $60$ km/h has a scalar speed of $60$. If we say the car is travelling at $60$ km/h due east, that is a vector (velocity). The number $60$ alone is the scalar magnitude.

**Example 3: Price.** A laptop costs $\$1{,}200$. The number $1200$ is a scalar. It represents the cost as a single quantity.

**Example 4: Age.** A person's age is $25$ years. This is a scalar — a single number on a continuous scale.

**Example 5: Mass.** A bag of rice has a mass of $5$ kilograms. The $5$ is a scalar.

## AI/ML Relevance

Scalars appear throughout the entire AI/ML pipeline. Here are concrete examples:

1. **Learning Rate ($\alpha$):** When training a neural network, the learning rate is a scalar (typically a small positive number like $0.001$ or $0.0001$) that controls step size during gradient descent. Choosing the right scalar value for the learning rate is one of the most important hyperparameter tuning decisions.

2. **Loss (Cost):** After a forward pass through a model, the loss function computes a single scalar that quantifies the error between predictions and ground truth. For example, Mean Squared Error (MSE) for a batch of $n$ predictions $\hat{y}_i$ and true values $y_i$ is:
   $$L = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2$$
   This $L$ is a scalar.

3. **Regularisation Strength ($\lambda$):** In L1 or L2 regularisation, a scalar $\lambda$ controls how much we penalise large weights. Setting $\lambda$ too high (e.g., $10.0$) causes underfitting; too low (e.g., $10^{-6}$) may cause overfitting.

4. **Evaluation Metrics:** Accuracy ($0.94$), precision ($0.87$), recall ($0.91$), and F1-score ($0.89$) are all scalars used to compare model performance.

5. **Gradient Magnitude:** The norm of a gradient vector is a scalar used to detect vanishing or exploding gradients.

6. **Softmax Temperature ($T$):** In knowledge distillation, a scalar temperature $T$ controls the softness of probability distributions. Higher $T$ produces softer distributions.

## Mathematical Explanation

A scalar is the simplest mathematical object in linear algebra. It belongs to a **field**, which is a set $F$ equipped with two operations — addition ($+$) and multiplication ($\cdot$) — that satisfy the following axioms for all $a, b, c \in F$:

1. **Closure under addition:** $a + b \in F$.
2. **Closure under multiplication:** $a \cdot b \in F$.
3. **Associativity of addition:** $(a + b) + c = a + (b + c)$.
4. **Associativity of multiplication:** $(a \cdot b) \cdot c = a \cdot (b \cdot c)$.
5. **Commutativity of addition:** $a + b = b + a$.
6. **Commutativity of multiplication:** $a \cdot b = b \cdot a$.
7. **Identity element for addition (zero):** There exists $0 \in F$ such that $a + 0 = a$.
8. **Identity element for multiplication (one):** There exists $1 \in F$ such that $a \cdot 1 = a$.
9. **Additive inverses:** For each $a$, there exists $-a \in F$ such that $a + (-a) = 0$.
10. **Multiplicative inverses:** For each $a \neq 0$, there exists $a^{-1} \in F$ such that $a \cdot a^{-1} = 1$.
11. **Distributivity:** $a \cdot (b + c) = (a \cdot b) + (a \cdot c)$.

In practice, the scalars used in machine learning are almost always **real numbers** ($\mathbb{R}$). Real numbers satisfy all of the above axioms. Occasionally, **complex numbers** ($\mathbb{C}$) are used (e.g., in signal processing or quantum machine learning), but real scalars are the default for most ML applications.

Scalars can be operated on with standard arithmetic:

- **Addition:** $a + b$ (e.g., $3 + 5 = 8$)
- **Subtraction:** $a - b$ (e.g., $10 - 4.5 = 5.5$)
- **Multiplication:** $a \times b$ or $a \cdot b$ (e.g., $7 \times 0.5 = 3.5$)
- **Division:** $a \div b$ or $a / b$ (e.g., $15 / 4 = 3.75$)
- **Exponentiation:** $a^b$ (e.g., $2^3 = 8$)
- **Square root:** $\sqrt{a}$ (e.g., $\sqrt{16} = 4$)

When a scalar multiplies a vector or matrix, it is called **scalar multiplication** — every component of the vector or matrix is multiplied by that scalar.

## Formula(s)

There is no single "formula for a scalar" because a scalar is defined by what it is rather than by an equation. However, the following formulas show how scalars interact with other mathematical objects:

**Scalar multiplication of a vector:**
$$c \cdot \mathbf{v} = c \cdot \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{pmatrix}$$

Where $c \in \mathbb{R}$ is a scalar and $\mathbf{v} \in \mathbb{R}^n$ is a vector.

**Scalar multiplication of a matrix:**
$$c \cdot \mathbf{A} = c \cdot \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} = \begin{pmatrix} c \cdot a_{11} & c \cdot a_{12} \\ c \cdot a_{21} & c \cdot a_{22} \end{pmatrix}$$

Where $c \in \mathbb{R}$ is a scalar and $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a matrix.

**Mean (a scalar computed from data):**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

This takes a set of numbers $x_1, x_2, \ldots, x_n$ and produces a single scalar $\bar{x}$.

**Variance (a scalar measuring spread):**
$$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

The variance $\sigma^2$ is a single scalar.

## Step-by-Step Worked Examples

### Example 1: Basic Scalar Operations

Suppose we need to compute the following expression involving scalars:

$$3 \times (5 + 2 \times 4) - 6 \div 2$$

**Step 1:** Evaluate inside the parentheses, following the order of operations (PEMDAS/BODMAS). Inside the parentheses, multiplication comes before addition:

$$5 + 2 \times 4 = 5 + 8 = 13$$

**Step 2:** Multiply the result by 3:

$$3 \times 13 = 39$$

**Step 3:** Perform the division:

$$6 \div 2 = 3$$

**Step 4:** Subtract:

$$39 - 3 = 36$$

**Answer:** $36$

### Example 2: Computing Mean Squared Error (MSE) Loss

We have three data points with true values $y = [3, 5, 2]$ and predictions $\hat{y} = [2.5, 5.0, 3.0]$. Compute the MSE loss, which is a scalar.

**Step 1:** Compute each error:

$$e_1 = \hat{y}_1 - y_1 = 2.5 - 3 = -0.5$$
$$e_2 = \hat{y}_2 - y_2 = 5.0 - 5 = 0.0$$
$$e_3 = \hat{y}_3 - y_3 = 3.0 - 2 = 1.0$$

**Step 2:** Square each error:

$$e_1^2 = (-0.5)^2 = 0.25$$
$$e_2^2 = (0.0)^2 = 0.00$$
$$e_3^2 = (1.0)^2 = 1.00$$

**Step 3:** Sum the squared errors:

$$0.25 + 0.00 + 1.00 = 1.25$$

**Step 4:** Divide by the number of data points $n = 3$:

$$L = \frac{1.25}{3} \approx 0.4167$$

**Answer:** The MSE loss is $0.4167$ (a scalar).

### Example 3: Scalar Multiplication of a Vector

A model's gradient vector is $\mathbf{g} = \begin{pmatrix} 4 \\ -2 \\ 0.5 \end{pmatrix}$. The learning rate is $\alpha = 0.01$. Compute the update step $\alpha \cdot \mathbf{g}$.

**Step 1:** Multiply each component of the vector by the scalar $\alpha = 0.01$:

Component 1: $0.01 \times 4 = 0.04$
Component 2: $0.01 \times (-2) = -0.02$
Component 3: $0.01 \times 0.5 = 0.005$

**Step 2:** Write the result as a vector:

$$\alpha \cdot \mathbf{g} = \begin{pmatrix} 0.04 \\ -0.02 \\ 0.005 \end{pmatrix}$$

**Answer:** $\begin{pmatrix} 0.04 \\ -0.02 \\ 0.005 \end{pmatrix}$

### Example 4: Computing a Weighted Sum (Scalar Output)

A simple linear model predicts house prices as $p = w_1 \cdot x_1 + w_2 \cdot x_2 + b$, where $w_1 = 3000$, $w_2 = 15000$, $b = 50000$, $x_1 = 3$ (bedrooms), and $x_2 = 0.15$ (acreage in hectares). Compute the predicted price $p$.

**Step 1:** Compute the contribution of bedrooms:

$$w_1 \cdot x_1 = 3000 \times 3 = 9000$$

**Step 2:** Compute the contribution of acreage:

$$w_2 \cdot x_2 = 15000 \times 0.15 = 2250$$

**Step 3:** Add the bias term:

$$p = 9000 + 2250 + 50000 = 61250$$

**Answer:** The predicted price is $\$61{,}250$ (a scalar).

### Example 5: Solving for a Scalar in an Equation

If $3(x - 4) + 2x = 28$, find the scalar $x$.

**Step 1:** Expand the brackets:

$$3x - 12 + 2x = 28$$

**Step 2:** Combine like terms:

$$5x - 12 = 28$$

**Step 3:** Add 12 to both sides:

$$5x = 40$$

**Step 4:** Divide both sides by 5:

$$x = 8$$

**Answer:** $x = 8$

## Visual Interpretation

A scalar can be visualised as a **point on a number line**.

```
Real Number Line:
<──|────|────|────|────|────|────|────|────|────|────|──>
  -4   -3   -2   -1    0    1    2    3    4    5    6
                                               ▲
                                               │
                                          Scalar: 4.5
```

Each scalar corresponds to exactly one location on the line. The number line is one-dimensional, reflecting the fact that a scalar has only magnitude (distance from zero) and no direction beyond being positive or negative.

In contrast, a vector in 2D would require a 2D plane for visualisation (an arrow with both length and direction), and a matrix would be represented as a grid of numbers. A scalar is fundamentally one-dimensional.

When a scalar multiplies a vector, it **stretches** or **shrinks** the vector without changing its direction (unless the scalar is negative, in which case the direction reverses). Imagine a rubber band: pulling it to double its length is like multiplying the length scalar by 2. Compressing it to half its length is like multiplying by $0.5$.

## Common Mistakes

1. **Confusing scalars with 1D vectors.** A common error is to treat a scalar like the number 5 as a vector containing one element, $[5]$. While related, they are different objects. A 1D vector has a position in a vector space and transforms under linear transformations; a scalar is invariant under such transformations. For example, if you rotate a coordinate system, a vector changes its components, but a scalar (like temperature) does not.

2. **Forgetting the order of operations.** In expressions like $3 + 4 \times 2$, beginners often compute $(3 + 4) \times 2 = 14$ instead of the correct $3 + (4 \times 2) = 11$. Always follow PEMDAS/BODMAS: Parentheses/Brackets, Exponents/Orders, Multiplication/Division (left to right), Addition/Subtraction (left to right).

3. **Misapplying scalar operations to vectors.** For example, trying to divide one vector by another directly ($\mathbf{v} / \mathbf{w}$) as if they were scalars. Vector division is not defined. Scalar division is only defined between two scalars, or between a vector and a scalar (component-wise).

4. **Thinking scalars must be positive.** Scalars can be any real number: positive, negative, or zero. Negative scalars are common (e.g., temperature of $-10^\circ$C, a loss function difference of $-0.5$ indicating improvement).

5. **Ignoring units when working with scalars.** Scalars often represent physical quantities with units. Computing $5$ metres $+ 3$ kilograms is meaningless. Always ensure scalar operations are performed on quantities with compatible units, or that the units are properly tracked.

6. **Confusing the learning rate scalar with the loss scalar.** During neural network training, the learning rate $\alpha$ (a scalar) controls step size, while the loss $L$ (also a scalar) measures error. Beginners sometimes mistakenly multiply these or confuse their roles.

7. **Assuming all scalars are real numbers.** While most scalars in ML are real numbers, in some contexts (quantum computing, complex-valued neural networks, signal processing) scalars can be complex numbers. The definition of a scalar as "a single number" still holds, but the number can be complex.

## Interview Questions

### Beginner

1. **What is a scalar? Give three examples from everyday life.**
   *Answer: A scalar is a single numerical value representing magnitude only. Examples: the temperature 30°C, the price $15.99, the mass 70 kg.*

2. **How is a scalar different from a vector?**
   *Answer: A scalar has only magnitude (e.g., 5 km/h speed), while a vector has both magnitude and direction (e.g., 5 km/h north).*

3. **Can a scalar be negative? Explain.**
   *Answer: Yes. Scalars can be any real number. For example, $-10^\circ$C is a valid scalar representing a temperature below freezing.*

4. **What happens when you multiply a scalar by a vector?**
   *Answer: Each component of the vector is multiplied by the scalar. If the scalar is $c$ and the vector is $\mathbf{v} = (v_1, v_2, \ldots, v_n)$, then $c\mathbf{v} = (c \cdot v_1, c \cdot v_2, \ldots, c \cdot v_n)$.*

5. **Is the mean of a set of numbers a scalar? Why?**
   *Answer: Yes. The mean is a single number computed from the dataset. It has no direction and represents a central tendency as a single value, making it a scalar.*

6. **What is the difference between $5$ (a scalar) and $[5]$ (a 1-element vector)?**
   *Answer: A scalar is a single number in a field. A 1-element vector is an ordered tuple containing one element, which lives in a 1-dimensional vector space. Under a change of coordinates, the scalar stays the same, while the vector's representation changes.*

### Intermediate

1. **Explain why the loss function output is always a scalar, even for multi-output models.**
   *Answer: A loss function like MSE or cross-entropy aggregates the errors across all outputs and all samples into a single number. This aggregation (summation, averaging) collapses the multi-dimensional error array into a scalar, which is necessary because optimisation algorithms (e.g., gradient descent) minimise a single objective value. Without a scalar loss, there would be no unambiguous direction of improvement.*

2. **How does the choice of a scalar learning rate affect convergence in gradient descent?**
   *Answer: A learning rate $\alpha$ that is too large can cause divergence (the loss oscillates or explodes because the steps overshoot the minimum). A learning rate that is too small converges very slowly or gets stuck in plateaus. An ideal learning rate balances convergence speed with stability. Adaptive methods like Adam adjust per-parameter scalars dynamically.*

3. **If a scalar $c$ is applied to a matrix $\mathbf{A}$ to produce $c\mathbf{A}$, what happens to the determinant?**
   *Answer: For an $n \times n$ matrix, $\det(c\mathbf{A}) = c^n \det(\mathbf{A})$. This is because multiplying every row by $c$ multiplies the determinant by $c$ for each row, so for $n$ rows the scaling factor is $c^n$. This shows that scalar multiplication of a matrix does not simply scale the determinant by $c$.*

4. **What does it mean for a scalar to be invariant under a change of basis?**
   *Answer: A scalar does not change when the coordinate system changes. For example, temperature is 30°C regardless of whether you use Cartesian or polar coordinates. In contrast, vector components change under basis transformations. This invariance is why scalars are sometimes called "rank-0 tensors".*

5. **In the softmax function $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$, what role does the scalar $T$ (temperature) play?**
   *Answer: The scalar $T$ controls the "sharpness" of the probability distribution. When $T=1$, the standard softmax is used. When $T > 1$, the distribution becomes softer (more uniform, better for exploration). When $T < 1$, the distribution becomes sharper (more confident, better for exploitation). As $T \to 0^+$, softmax approaches a one-hot argmax. As $T \to \infty$, it approaches a uniform distribution.*

6. **Why does adding L2 regularisation $\lambda \sum w_i^2$ to a loss function produce a scalar adjustment?**
   *Answer: The regularisation term $\lambda \sum w_i^2$ is itself a scalar: $\lambda$ is a scalar hyperparameter, and $\sum w_i^2$ is the sum of squared scalar weights, which is also a scalar. Adding two scalars (the data loss and the regularisation loss) produces another scalar. This combined scalar is what gradient descent minimises.*

### Advanced

1. **In differential geometry, a scalar field on a manifold is defined as a function $f: M \to \mathbb{R}$ that is invariant under coordinate transformations. Explain why this definition generalises the elementary notion of a scalar.**
   *Answer: In elementary contexts, a scalar is just a number. In differential geometry, a scalar field assigns a scalar to every point on a manifold, and the key property is coordinate invariance: if you change coordinates, $f(p)$ at point $p$ remains the same number. This matches the elementary property that a scalar does not depend on the basis or coordinate system. The generalisation allows us to talk about scalars on curved spaces (e.g., the temperature at every point on Earth's surface).*

2. **Consider a neural network with loss $L$. The gradient $\nabla L$ is a vector. The directional derivative $\nabla_\mathbf{v} L = \nabla L \cdot \mathbf{v}$ is a scalar. Explain the geometric meaning of this scalar and how it is used in optimisation.**
   *Answer: The directional derivative $\nabla L \cdot \mathbf{v}$ is the scalar that tells us the instantaneous rate of change of $L$ when moving in direction $\mathbf{v}$. If we choose $\mathbf{v} = -\nabla L / \|\nabla L\|$ (the negative normalised gradient), the directional derivative is $-\|\nabla L\|$, which is the steepest descent direction. This scalar is the slope of the loss landscape in that direction. In optimisation, we want to make this scalar as negative as possible to reduce loss the fastest.*

3. **In Bayesian machine learning, the evidence $p(D)$ (also called the marginal likelihood) is a scalar computed as $p(D) = \int p(D|\theta) p(\theta) \, d\theta$. Why is this scalar important, and what practical challenges arise in computing it?**
   *Answer: The evidence scalar is crucial for model selection — it tells us how likely the observed data is under a given model, integrating over all possible parameters. A higher evidence indicates a better model. However, computing the integral exactly is intractable for high-dimensional $\theta$. Techniques like variational inference (which maximises the ELBO, a tractable scalar bound) or MCMC are used to approximate it. The fact that the evidence is a scalar allows direct comparison between models via Bayes factors.*

## Practice Problems

### Easy

1. Compute $15 - 3 \times 4 + 2$.

2. If $x = 7$ and $y = 3$, compute $2x + 3y - 5$.

3. Multiply the scalar $c = -2$ by the vector $\mathbf{v} = \begin{pmatrix} 3 \\ -1 \\ 4 \end{pmatrix}$.

4. The true value is $y = 10$ and the prediction is $\hat{y} = 7.5$. Compute the squared error $(y - \hat{y})^2$.

5. A model's accuracy on a test set is $0.92$. Express this accuracy as a percentage.

### Medium

1. Compute $\frac{3}{4} \times (16 - 4 \times 2) + \sqrt{25}$.

2. A linear regression model predicts salary as $\text{salary} = 30000 + 5000 \times \text{years\_experience}$. For 8 years of experience, what is the predicted salary (a scalar)? If the actual salary is $72000$, what is the absolute error (also a scalar)?

3. The loss function $L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ is given data: $y = [1, 4, 3]$, $\hat{y} = [1.5, 3.5, 3.0]$. Compute $L$.

4. If scalar $a = 2$ and $\mathbf{A} = \begin{pmatrix} 1 & 3 \\ -2 & 0 \end{pmatrix}$, compute $a\mathbf{A}$ and then find the determinant of the resulting matrix (hint: $\det(a\mathbf{A}) = a^n \det(\mathbf{A})$ for an $n \times n$ matrix). Verify by direct computation.

5. Solve for scalar $x$: $4(2x - 1) + 3 = 2x + 17$.

### Hard

1. In gradient descent, the update rule is $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$. Suppose $\mathbf{w}_t = \begin{pmatrix} 1 \\ -2 \end{pmatrix}$, $\nabla L(\mathbf{w}_t) = \begin{pmatrix} 3 \\ -4 \end{pmatrix}$, and $\alpha = 0.5$. Compute $\mathbf{w}_{t+1}$. What scalar is the step size? What vector is the step direction?

2. Consider the function $f(x) = 3x^2 - 12x + 7$. Find the scalar $x$ that minimises $f$. (Hint: take the derivative, set to zero, solve for $x$.) Then compute the minimum value of $f$ (also a scalar).

3. A machine learning model has a loss function with an L2 regularisation term: $L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p w_j^2$. Given $n=2$, $p=2$, $y = [2, 4]$, $\hat{y} = [1, 5]$, $w = [0.5, -1.0]$, and $\lambda = 0.1$, compute the scalar $L$.

## Solutions

### Easy Solutions

**1.** $15 - 3 \times 4 + 2 = 15 - 12 + 2 = 5$

**2.** $2x + 3y - 5 = 2(7) + 3(3) - 5 = 14 + 9 - 5 = 18$

**3.** $c\mathbf{v} = -2 \cdot \begin{pmatrix} 3 \\ -1 \\ 4 \end{pmatrix} = \begin{pmatrix} -6 \\ 2 \\ -8 \end{pmatrix}$

**4.** $(y - \hat{y})^2 = (10 - 7.5)^2 = (2.5)^2 = 6.25$

**5.** $0.92 \times 100 = 92\%$

### Medium Solutions

**1.** $\frac{3}{4} \times (16 - 4 \times 2) + \sqrt{25} = \frac{3}{4} \times (16 - 8) + 5 = \frac{3}{4} \times 8 + 5 = 6 + 5 = 11$

**2.** Predicted salary: $30000 + 5000 \times 8 = 30000 + 40000 = 70000$. Absolute error: $|72000 - 70000| = 2000$.

**3.** Compute errors: $y_1 - \hat{y}_1 = 1 - 1.5 = -0.5$, $y_2 - \hat{y}_2 = 4 - 3.5 = 0.5$, $y_3 - \hat{y}_3 = 3 - 3.0 = 0.0$.
Squared errors: $(-0.5)^2 = 0.25$, $(0.5)^2 = 0.25$, $(0.0)^2 = 0.0$.
Mean: $L = (0.25 + 0.25 + 0.0) / 3 = 0.5 / 3 \approx 0.1667$.

**4.** $a\mathbf{A} = 2 \cdot \begin{pmatrix} 1 & 3 \\ -2 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 6 \\ -4 & 0 \end{pmatrix}$.
$\det(a\mathbf{A}) = (2)(0) - (6)(-4) = 0 + 24 = 24$.
Verification using the formula: $\det(a\mathbf{A}) = a^2 \det(\mathbf{A}) = 4 \times ((1)(0) - (3)(-2)) = 4 \times (0 + 6) = 4 \times 6 = 24$. ✓

**5.** $4(2x - 1) + 3 = 2x + 17$. Expand: $8x - 4 + 3 = 2x + 17$. Simplify: $8x - 1 = 2x + 17$. Subtract $2x$: $6x - 1 = 17$. Add 1: $6x = 18$. Divide: $x = 3$.

### Hard Solutions

**1.** $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t) = \begin{pmatrix} 1 \\ -2 \end{pmatrix} - 0.5 \cdot \begin{pmatrix} 3 \\ -4 \end{pmatrix} = \begin{pmatrix} 1 \\ -2 \end{pmatrix} - \begin{pmatrix} 1.5 \\ -2 \end{pmatrix} = \begin{pmatrix} -0.5 \\ 0 \end{pmatrix}$.
The step size is the scalar $\alpha \cdot \|\nabla L\| = 0.5 \times \sqrt{3^2 + (-4)^2} = 0.5 \times 5 = 2.5$.
The step direction is the normalised gradient $\frac{\nabla L}{\|\nabla L\|} = \begin{pmatrix} 3/5 \\ -4/5 \end{pmatrix} = \begin{pmatrix} 0.6 \\ -0.8 \end{pmatrix}$.

**2.** Derivative: $f'(x) = 6x - 12$. Set to zero: $6x - 12 = 0 \Rightarrow x = 2$.
Minimum value: $f(2) = 3(2)^2 - 12(2) + 7 = 12 - 24 + 7 = -5$.

**3.** Step 1 — Data loss: $n=2$, errors are $2-1=1$ and $4-5=-1$. Squared errors: $1^2=1$, $(-1)^2=1$. Mean: $(1+1)/2 = 1$.
Step 2 — Regularisation term: $\sum w_j^2 = (0.5)^2 + (-1.0)^2 = 0.25 + 1.0 = 1.25$. With $\lambda=0.1$: $0.1 \times 1.25 = 0.125$.
Step 3 — Total: $L = 1 + 0.125 = 1.125$.

## Related Concepts

- **Number Systems** — Scalars belong to number systems (natural numbers, integers, rationals, reals, complex numbers). Understanding number systems clarifies what kinds of scalars exist.
- **Real Numbers** — The most common type of scalar used in AI/ML is a real number.
- **Complex Numbers** — A generalisation of real scalars used in advanced ML contexts.
- **Mathematical Notation** — The symbols and conventions used to write scalars and scalar expressions.

## Next Concepts

- **Vector** — A vector is an ordered collection of scalars. Understanding scalars is prerequisite to understanding vectors because vector components are scalars and scalar multiplication is a fundamental vector operation.
- **Matrix** — A matrix is a rectangular array of scalars. Matrix operations (addition, multiplication, scalar multiplication) all rely on scalar arithmetic.
- **Tensor** — A tensor is a multi-dimensional generalisation where scalars are rank-0 tensors, vectors are rank-1, and matrices are rank-2.

## Summary

A scalar is a single numerical value representing magnitude only, with no direction. Scalars are the simplest mathematical objects in linear algebra and serve as the building blocks for vectors, matrices, and tensors. They support all standard arithmetic operations and are used throughout AI/ML for learning rates, loss values, regularisation parameters, evaluation metrics, and more. A solid grasp of scalars and scalar operations is essential for progressing to more advanced mathematical concepts in machine learning.

## Key Takeaways

- A scalar is a single number with magnitude only — no direction.
- Scalars can be positive, negative, or zero, and are typically real numbers in ML contexts.
- Scalars are the fundamental building blocks of vectors, matrices, and tensors.
- In AI/ML, scalars appear as learning rates, loss values, regularisation strengths, accuracy metrics, and gradient magnitudes.
- All arithmetic operations (addition, subtraction, multiplication, division) apply to scalars following standard algebraic rules.
- A scalar is invariant under coordinate transformations — it does not change when the basis or coordinate system changes.
