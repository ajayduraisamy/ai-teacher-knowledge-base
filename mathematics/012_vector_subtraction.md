# Concept: Vector Subtraction

## Concept ID

MATH-012

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Subtract one vector from another using component-wise subtraction.
- Interpret vector subtraction geometrically as the vector from one point to another.
- Understand vector subtraction in terms of additive inverses: $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$.
- Apply vector subtraction in AI/ML contexts such as computing error vectors, deltas, and differences.
- Solve real-world problems involving vector differences.

## Prerequisites

- **Vector (MATH-002):** Understanding that a vector has magnitude and direction.
- **Vector Addition (MATH-011):** The component-wise addition operation and its properties.
- **Scalar (MATH-001):** Basic arithmetic with real numbers, including handling negative numbers.
- **Coordinate System (MATH-006):** Familiarity with coordinates on a plane.

## Definition

**Vector subtraction** is the operation of finding the difference between two vectors. If $\mathbf{u}$ and $\mathbf{v}$ are vectors of the same dimension $n$, their difference $\mathbf{u} - \mathbf{v}$ is computed by subtracting corresponding components:

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} u_1 - v_1 \\ u_2 - v_2 \\ \vdots \\ u_n - v_n \end{pmatrix}$$

The result is a vector of the same dimension. Vector subtraction is equivalent to adding the **additive inverse** of $\mathbf{v}$:

$$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$$

where $-\mathbf{v} = (-v_1, -v_2, \ldots, -v_n)^T$.

## Intuition

Vector subtraction answers the question: **What vector takes me from one point to another?**

If you are at point $A$ (represented by position vector $\mathbf{a}$) and you want to get to point $B$ (represented by $\mathbf{b}$), the vector from $A$ to $B$ is:

$$\overrightarrow{AB} = \mathbf{b} - \mathbf{a}$$

This is the most intuitive interpretation of vector subtraction: it gives the **displacement** from one vector to another.

Another way to think about it: if adding $\mathbf{v}$ to $\mathbf{u}$ means "apply $\mathbf{v}$ after $\mathbf{u}$", then subtracting $\mathbf{v}$ means "undo $\mathbf{v}$" — it is the inverse operation. If $\mathbf{u} + \mathbf{v} = \mathbf{w}$, then $\mathbf{w} - \mathbf{v} = \mathbf{u}$.

In AI/ML terms, vector subtraction is how we compute **errors**: if $\mathbf{y}$ is the true target vector and $\hat{\mathbf{y}}$ is the predicted vector, then $\mathbf{y} - \hat{\mathbf{y}}$ is the error vector, telling us how far off the prediction is in each component.

## Why This Concept Matters

Vector subtraction is essential for measuring differences between multi-dimensional quantities. Without it, we could not compute errors, distances, or deltas between any two points in vector space.

In AI and machine learning, vector subtraction appears in:
- **Loss computation:** The error vector $\mathbf{y} - \hat{\mathbf{y}}$ is the starting point for most loss functions (MSE, MAE, cross-entropy with softmax).
- **Gradient descent updates:** The parameter update is $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L$, which involves subtracting the scaled gradient from the current weights.
- **Word embedding analogies:** $\text{vector("king")} - \text{vector("man")} + \text{vector("woman")} \approx \text{vector("queen")}$ — the subtraction isolates the "masculinity" direction.
- **Computing displacement vectors:** Any "vector from $A$ to $B$" is $\mathbf{b} - \mathbf{a}$, which is used in spatial reasoning, robotics, and computer graphics.

Without vector subtraction, we could not measure how different two vectors are, and most of machine learning (which is fundamentally about minimising differences) would be impossible.

## Historical Background

Vector subtraction developed alongside vector addition in the 19th century. The key insight — that subtraction is just addition of the negative — was formalised as part of the abstract definition of a vector space.

In physics, vector subtraction had been used implicitly for centuries to compute relative velocities and relative positions. For example, if a boat moves at velocity $\mathbf{v}_b$ relative to the water and the water moves at $\mathbf{v}_w$ relative to the ground, the boat's velocity relative to the ground is $\mathbf{v}_b + \mathbf{v}_w$. To find the boat's velocity relative to the water given its ground velocity, you subtract: $\mathbf{v}_{\text{boat rel. water}} = \mathbf{v}_{\text{boat rel. ground}} - \mathbf{v}_{\text{water rel. ground}}$.

The algebraic treatment of subtraction as "adding the additive inverse" was solidified in the early 20th century with the axiomatisation of vector spaces by Giuseppe Peano (1888) and later by Hermann Weyl and others.

## Real World Examples

**Example 1: Relative position.** A car is at position $\mathbf{a} = (3, 5)$ km and a truck is at $\mathbf{b} = (7, 2)$ km relative to the same origin. The vector from the car to the truck is $\mathbf{b} - \mathbf{a} = (4, -3)$ km — the truck is 4 km east and 3 km south of the car.

**Example 2: Relative velocity.** An aeroplane's ground velocity is $\mathbf{v}_g = (250, 30)$ m/s. The wind velocity is $\mathbf{v}_w = (20, -5)$ m/s. The aeroplane's airspeed (velocity relative to the air) is $\mathbf{v}_a = \mathbf{v}_g - \mathbf{v}_w = (230, 35)$ m/s.

**Example 3: Force difference.** Two forces act on an object: $\mathbf{F}_1 = (10, 5)$ N and $\mathbf{F}_2 = (7, 9)$ N. The additional force needed to make $\mathbf{F}_1$ equal to $\mathbf{F}_2$ is $\mathbf{F}_2 - \mathbf{F}_1 = (-3, 4)$ N.

**Example 4: GPS navigation.** A hiker's current position is $\mathbf{p}_{\text{current}} = (4, 7)$ km from base camp. The destination is $\mathbf{p}_{\text{dest}} = (10, 3)$ km. The displacement vector the hiker must travel is $\mathbf{d} = \mathbf{p}_{\text{dest}} - \mathbf{p}_{\text{current}} = (6, -4)$ km.

## AI/ML Relevance

Vector subtraction is a fundamental operation in nearly every machine learning algorithm:

1. **Computing Error Vectors.** The most common use of vector subtraction in ML is computing the difference between predictions and targets. For a batch of predictions $\hat{\mathbf{y}}$ and true values $\mathbf{y}$:

   $$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$$

   This error vector is then used to compute the loss. For Mean Squared Error (MSE):

   $$L = \frac{1}{n} \|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

   The entire learning process is about minimising this difference vector.

2. **Gradient Descent Updates.** The core update rule for gradient descent involves subtracting a vector from the parameter vector:

   $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

   Here, $\alpha \nabla L$ (the scaled gradient) is subtracted from the current weights. This moves the parameters in the direction that reduces the loss.

3. **Word Embedding Differences.** In NLP, vector subtraction reveals semantic relationships:

   $$\text{vector("France")} - \text{vector("Paris")} \approx \text{vector("Italy")} - \text{vector("Rome")}$$

   The difference $\text{vector("France")} - \text{vector("Paris")}$ captures the "country minus capital" direction. This works because the embedding space encodes relational structure as vector offsets.

4. **Change Detection.** In time series or anomaly detection, the difference between consecutive feature vectors $\mathbf{x}_{t+1} - \mathbf{x}_t$ captures the change between time steps. Large differences may indicate anomalies.

5. **Contrastive Learning.** In self-supervised learning, the loss function often pulls similar vectors together and pushes dissimilar vectors apart by operating on vector differences:

   $$L = \|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2 \quad \text{for similar pairs}$$

   The subtraction $f(\mathbf{x}_i) - f(\mathbf{x}_j)$ gives the vector difference, and its norm measures how far apart the representations are.

6. **Policy Gradient Updates.** In reinforcement learning, the advantage function $A(s, a) = Q(s, a) - V(s)$ is a scalar computed from vector-value differences, used to update the policy.

## Mathematical Explanation

### Component-Wise Subtraction

Given two vectors $\mathbf{u}$ and $\mathbf{v}$ of the same dimension, their difference is:

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} u_1 - v_1 \\ u_2 - v_2 \\ \vdots \\ u_n - v_n \end{pmatrix}$$

### Relationship to Addition

Vector subtraction is defined in terms of addition:

$$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$$

where $-\mathbf{v}$ is the additive inverse of $\mathbf{v}$ (each component negated). This means all the algebraic properties of addition (commutativity, associativity) apply indirectly, but note that subtraction itself is **not** commutative: $\mathbf{u} - \mathbf{v} \neq \mathbf{v} - \mathbf{u}$ in general. Instead, $\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$.

### Geometric Interpretation

Geometrically, $\mathbf{u} - \mathbf{v}$ is the vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$ when both vectors are drawn from the same starting point.

1. Draw $\mathbf{u}$ and $\mathbf{v}$ starting from the same origin.
2. The difference $\mathbf{u} - \mathbf{v}$ is the vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$.

This is why $\mathbf{b} - \mathbf{a}$ gives the vector from $A$ to $B$: if $\mathbf{a}$ and $\mathbf{b}$ are position vectors of points $A$ and $B$, then $\mathbf{b} - \mathbf{a}$ points from $A$ to $B$.

### Alternative Interpretation: Reversing Addition

If $\mathbf{u} + \mathbf{v} = \mathbf{w}$, then:
- $\mathbf{w} - \mathbf{v} = \mathbf{u}$ (undoing the addition of $\mathbf{v}$)
- $\mathbf{w} - \mathbf{u} = \mathbf{v}$ (undoing the addition of $\mathbf{u}$)

This is exactly how subtraction works with ordinary numbers: if $3 + 5 = 8$, then $8 - 5 = 3$ and $8 - 3 = 5$.

### Subtraction of Multiple Vectors

Subtracting multiple vectors is done component-wise:

$$\mathbf{u} - \mathbf{v} - \mathbf{w} = \begin{pmatrix} u_1 - v_1 - w_1 \\ u_2 - v_2 - w_2 \\ \vdots \\ u_n - v_n - w_n \end{pmatrix}$$

This is equivalent to $\mathbf{u} + (-\mathbf{v}) + (-\mathbf{w})$.

## Formula(s)

**Component-wise vector subtraction:**

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} u_1 - v_1 \\ u_2 - v_2 \\ \vdots \\ u_n - v_n \end{pmatrix}$$

**Difference as additive inverse:**

$$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$$

**Vector from point $A$ to point $B$:**

$$\overrightarrow{AB} = \mathbf{b} - \mathbf{a}$$

## Properties

Let $\mathbf{u}, \mathbf{v}, \mathbf{w} \in \mathbb{R}^n$ be vectors.

| Property | Formula |
|---|---|
| **Not commutative** | $\mathbf{u} - \mathbf{v} \neq \mathbf{v} - \mathbf{u}$ (unless $\mathbf{u} = \mathbf{v}$) |
| **Anti-commutative** | $\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$ |
| **Self-subtraction** | $\mathbf{v} - \mathbf{v} = \mathbf{0}$ |
| **Subtracting zero** | $\mathbf{v} - \mathbf{0} = \mathbf{v}$ |
| **Zero minus vector** | $\mathbf{0} - \mathbf{v} = -\mathbf{v}$ |
| **Relation to addition** | $(\mathbf{u} + \mathbf{v}) - \mathbf{w} = \mathbf{u} + (\mathbf{v} - \mathbf{w})$ |
| **Distributive** | $c(\mathbf{u} - \mathbf{v}) = c\mathbf{u} - c\mathbf{v}$ |

The anti-commutative property is important: swapping the order negates the result. Geometrically, the vector from $B$ to $A$ is the negative of the vector from $A$ to $B$: $\overrightarrow{BA} = -\overrightarrow{AB}$.

## Step-by-Step Worked Examples

### Example 1: Basic 2D Vector Subtraction

Let $\mathbf{u} = \begin{pmatrix} 7 \\ -2 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 3 \\ 5 \end{pmatrix}$. Compute $\mathbf{u} - \mathbf{v}$ and $\mathbf{v} - \mathbf{u}$.

**Step 1:** Compute $\mathbf{u} - \mathbf{v}$ by subtracting corresponding components:
$$(\mathbf{u} - \mathbf{v})_1 = 7 - 3 = 4$$
$$(\mathbf{u} - \mathbf{v})_2 = -2 - 5 = -7$$

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} 4 \\ -7 \end{pmatrix}$$

**Step 2:** Compute $\mathbf{v} - \mathbf{u}$:
$$(\mathbf{v} - \mathbf{u})_1 = 3 - 7 = -4$$
$$(\mathbf{v} - \mathbf{u})_2 = 5 - (-2) = 7$$

$$\mathbf{v} - \mathbf{u} = \begin{pmatrix} -4 \\ 7 \end{pmatrix}$$

**Step 3:** Verify anti-commutativity:
$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} 4 \\ -7 \end{pmatrix} = -\begin{pmatrix} -4 \\ 7 \end{pmatrix} = -(\mathbf{v} - \mathbf{u})$$

**Answer:** $\mathbf{u} - \mathbf{v} = \begin{pmatrix}4 & -7\end{pmatrix}^T$, $\mathbf{v} - \mathbf{u} = \begin{pmatrix}-4 & 7\end{pmatrix}^T$.

### Example 2: Computing the Error Vector in Regression

A linear regression model predicts house prices. For 3 houses, the true prices and predictions (in $1000s) are:

$$\mathbf{y} = \begin{pmatrix} 350 \\ 280 \\ 410 \end{pmatrix}, \quad \hat{\mathbf{y}} = \begin{pmatrix} 340 \\ 295 \\ 400 \end{pmatrix}$$

Compute the error vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$, then compute the Mean Squared Error (MSE) $L = \frac{1}{3}\|\mathbf{e}\|^2$.

**Step 1:** Compute the error vector component-wise:
$$e_1 = y_1 - \hat{y}_1 = 350 - 340 = 10$$
$$e_2 = y_2 - \hat{y}_2 = 280 - 295 = -15$$
$$e_3 = y_3 - \hat{y}_3 = 410 - 400 = 10$$

$$\mathbf{e} = \begin{pmatrix} 10 \\ -15 \\ 10 \end{pmatrix}$$

**Step 2:** Compute the squared norm (sum of squared errors):
$$\|\mathbf{e}\|^2 = 10^2 + (-15)^2 + 10^2 = 100 + 225 + 100 = 425$$

**Step 3:** Divide by the number of samples:
$$L = \frac{425}{3} \approx 141.67$$

**Answer:** The error vector is $\mathbf{e} = \begin{pmatrix}10 & -15 & 10\end{pmatrix}^T$ and the MSE is $141.67$.

### Example 3: Finding the Vector Between Two Points

Point $A$ is at coordinates $(1, 4)$ and point $B$ is at $(5, 2)$. Find the vector $\overrightarrow{AB}$ and $\overrightarrow{BA}$.

**Step 1:** $\overrightarrow{AB}$ is the vector from $A$ to $B$:
$$\overrightarrow{AB} = \mathbf{b} - \mathbf{a} = \begin{pmatrix} 5 - 1 \\ 2 - 4 \end{pmatrix} = \begin{pmatrix} 4 \\ -2 \end{pmatrix}$$

This means: to go from $A$ to $B$, move 4 units right and 2 units down.

**Step 2:** $\overrightarrow{BA}$ is the vector from $B$ to $A$:
$$\overrightarrow{BA} = \mathbf{a} - \mathbf{b} = \begin{pmatrix} 1 - 5 \\ 4 - 2 \end{pmatrix} = \begin{pmatrix} -4 \\ 2 \end{pmatrix}$$

**Step 3:** Verify: $\overrightarrow{AB} = -\overrightarrow{BA}$:
$$\begin{pmatrix} 4 \\ -2 \end{pmatrix} = -\begin{pmatrix} -4 \\ 2 \end{pmatrix}$$

**Answer:** $\overrightarrow{AB} = \begin{pmatrix}4 & -2\end{pmatrix}^T$, $\overrightarrow{BA} = \begin{pmatrix}-4 & 2\end{pmatrix}^T$.

### Example 4: Word Embedding Analogy

Given the following word embedding vectors (simplified to 2D for illustration):

$$\mathbf{w}_{\text{king}} = \begin{pmatrix} 0.8 \\ 0.6 \end{pmatrix}, \quad
\mathbf{w}_{\text{man}} = \begin{pmatrix} 0.7 \\ 0.5 \end{pmatrix}, \quad
\mathbf{w}_{\text{woman}} = \begin{pmatrix} 0.5 \\ 0.7 \end{pmatrix}$$

Compute $\mathbf{w}_{\text{king}} - \mathbf{w}_{\text{man}} + \mathbf{w}_{\text{woman}}$ and compare it to $\mathbf{w}_{\text{queen}} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}$.

**Step 1:** Compute $\mathbf{w}_{\text{king}} - \mathbf{w}_{\text{man}}$ (the "masculinity" direction):
$$\mathbf{w}_{\text{king}} - \mathbf{w}_{\text{man}} = \begin{pmatrix} 0.8 - 0.7 \\ 0.6 - 0.5 \end{pmatrix} = \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix}$$

**Step 2:** Add $\mathbf{w}_{\text{woman}}$:
$$\mathbf{w}_{\text{king}} - \mathbf{w}_{\text{man}} + \mathbf{w}_{\text{woman}} = \begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix} + \begin{pmatrix} 0.5 \\ 0.7 \end{pmatrix} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}$$

**Step 3:** Compare with $\mathbf{w}_{\text{queen}}$:
$$\text{result} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix} = \mathbf{w}_{\text{queen}}$$

**Answer:** The result exactly matches the queen vector, demonstrating how vector subtraction captures semantic relationships. The subtraction $\mathbf{w}_{\text{king}} - \mathbf{w}_{\text{man}}$ isolates the "royalty" offset, which when added to "woman" gives "queen".

### Example 5: Gradient Descent Parameter Update

A model has weights $\mathbf{w} = \begin{pmatrix} 2.0 \\ -1.5 \\ 0.5 \end{pmatrix}$. The gradient of the loss with respect to these weights is $\nabla L = \begin{pmatrix} 0.8 \\ -0.3 \\ 0.6 \end{pmatrix}$. The learning rate is $\alpha = 0.1$. Compute the updated weights $\mathbf{w}_{\text{new}} = \mathbf{w} - \alpha \nabla L$.

**Step 1:** Compute the scaled gradient:
$$\alpha \nabla L = 0.1 \times \begin{pmatrix} 0.8 \\ -0.3 \\ 0.6 \end{pmatrix} = \begin{pmatrix} 0.08 \\ -0.03 \\ 0.06 \end{pmatrix}$$

**Step 2:** Subtract from the current weights:
$$\mathbf{w}_{\text{new}} = \begin{pmatrix} 2.0 \\ -1.5 \\ 0.5 \end{pmatrix} - \begin{pmatrix} 0.08 \\ -0.03 \\ 0.06 \end{pmatrix} = \begin{pmatrix} 1.92 \\ -1.47 \\ 0.44 \end{pmatrix}$$

**Answer:** The updated weights are $\begin{pmatrix}1.92 & -1.47 & 0.44\end{pmatrix}^T$. Note that the second weight increased (from $-1.5$ to $-1.47$) because the gradient component was negative, and subtracting a negative is addition.

## Visual Interpretation

Geometrically, $\mathbf{u} - \mathbf{v}$ is the vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$:

```
y
▲
│        u
│      ──▶
│     ╱│
│    ╱ │
│   ╱  │ u-v
│  ╱   │
│ ╱    ▼
│v
└───────────▶ x
```

Draw $\mathbf{u}$ and $\mathbf{v}$ from the same origin. The vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$ is $\mathbf{u} - \mathbf{v}$.

Alternatively, think of it as:
1. Draw $\mathbf{u}$ as an arrow from the origin.
2. Draw $-\mathbf{v}$ (the same length as $\mathbf{v}$ but pointing opposite direction) from the tip of $\mathbf{u}$.
3. The sum $\mathbf{u} + (-\mathbf{v})$ is the vector from the origin to the final tip.

This visualises why $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$.

## Common Mistakes

1. **Subtracting in the wrong order.** The vector from $A$ to $B$ is $\mathbf{b} - \mathbf{a}$, not $\mathbf{a} - \mathbf{b}$. Always remember: tip minus tail. The point you want to go **to** minus the point you are **at**.

2. **Forgetting that subtraction is not commutative.** $\mathbf{u} - \mathbf{v}$ and $\mathbf{v} - \mathbf{u}$ are different vectors — they point in opposite directions. $\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$.

3. **Subtracting vectors of different dimensions.** Like addition, subtraction is only defined for vectors of the same dimension. $\begin{pmatrix}1 \\ 2\end{pmatrix} - \begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix}$ is undefined.

4. **Confusing vector subtraction with scalar subtraction of magnitudes.** The magnitude of the difference $\|\mathbf{u} - \mathbf{v}\|$ is not the same as $\|\mathbf{u}\| - \|\mathbf{v}\|$. For example, $\mathbf{u} = (3,4)$ has magnitude $5$ and $\mathbf{v} = (-3,-4)$ has magnitude $5$, but $\|\mathbf{u} - \mathbf{v}\| = \|(6,8)\| = 10$, not $0$.

5. **Treating the error vector as a scalar.** The error vector $\mathbf{y} - \hat{\mathbf{y}}$ has one component per dimension. Beginners sometimes mistakenly compute a single scalar error instead of a vector. You need the vector to understand which dimensions have high error.

6. **Ignoring the sign of components.** A negative component in $\mathbf{u} - \mathbf{v}$ means $\mathbf{u}$ has less of that component than $\mathbf{v}$. In error analysis, a negative error means the model over-predicted that component (prediction was too high).

7. **Thinking subtraction is just addition with a minus sign but not negating all components.** Remember: $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$, and $-\mathbf{v}$ means **every** component is negated, not just the first one.

## Interview Questions

### Beginner

1. **How is vector subtraction defined?**
   *Answer: Vector subtraction is performed component-wise: $(\mathbf{u} - \mathbf{v})_i = u_i - v_i$ for each component $i$. Both vectors must have the same dimension.*

2. **What is the geometric meaning of $\mathbf{u} - \mathbf{v}$ when both vectors start at the same point?**
   *Answer: It is the vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$. It tells you what displacement is needed to go from the endpoint of $\mathbf{v}$ to the endpoint of $\mathbf{u}$.*

3. **If $\mathbf{a}$ is the position vector of point $A$ and $\mathbf{b}$ is the position vector of point $B$, what is $\overrightarrow{AB}$?**
   *Answer: $\overrightarrow{AB} = \mathbf{b} - \mathbf{a}$. It is the vector from $A$ to $B$.*

4. **What is $\mathbf{v} - \mathbf{v}$ equal to?**
   *Answer: $\mathbf{v} - \mathbf{v} = \mathbf{0}$, the zero vector. Every component cancels out because $v_i - v_i = 0$ for each $i$.*

5. **Is there a relationship between $\mathbf{u} - \mathbf{v}$ and $\mathbf{v} - \mathbf{u}$?**
   *Answer: Yes. $\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$. They are negatives of each other. Swapping the order reverses the direction of the result.*

### Intermediate

1. **Explain why vector subtraction is equivalent to adding the additive inverse. How does this help with algebraic manipulation?**
   *Answer: $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$, where $-\mathbf{v} = (-v_1, \ldots, -v_n)^T$. This equivalence means we can apply all the properties of addition (commutativity, associativity) to expressions involving subtraction by first converting subtraction to addition of negatives. For example, $(\mathbf{u} - \mathbf{v}) + \mathbf{w} = \mathbf{u} + (-\mathbf{v}) + \mathbf{w} = \mathbf{u} + \mathbf{w} - \mathbf{v}$.*

2. **In machine learning, why do we compute $\mathbf{y} - \hat{\mathbf{y}}$ rather than $\hat{\mathbf{y}} - \mathbf{y}$?**
   *Answer: The convention $\mathbf{y} - \hat{\mathbf{y}}$ gives the error vector that, when added to the prediction, yields the true value: $\hat{\mathbf{y}} + (\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{y}$. If we used $\hat{\mathbf{y}} - \mathbf{y}$, the error would be the negative of this. In gradient descent, we typically want to move predictions toward targets, so the sign convention matters. Either can be used as long as the optimisation is adjusted accordingly, but $\mathbf{y} - \hat{\mathbf{y}}$ is the standard convention.*

3. **How does the anti-commutative property $\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$ manifest in the real world?**
   *Answer: If the vector from city $A$ to city $B$ is $\overrightarrow{AB} = (100, 50)$ km, then the vector from $B$ to $A$ is $\overrightarrow{BA} = -\overrightarrow{AB} = (-100, -50)$ km. The distance is the same, but the direction is opposite. This is obvious in navigation but is captured mathematically by the anti-commutative property.*

4. **What is the difference between computing $\|\mathbf{y} - \hat{\mathbf{y}}\|$ and computing $\|\mathbf{y}\| - \|\hat{\mathbf{y}}\|$? Give an example where they differ.**
   *Answer: $\|\mathbf{y} - \hat{\mathbf{y}}\|$ computes the Euclidean distance between the two vectors, which accounts for both differences in magnitude and direction. $\|\mathbf{y}\| - \|\hat{\mathbf{y}}\|$ is just the difference of their magnitudes, which loses all directional information. Example: $\mathbf{y} = (1,0)$, $\hat{\mathbf{y}} = (-1,0)$. Then $\|\mathbf{y} - \hat{\mathbf{y}}\| = \|(2,0)\| = 2$, but $\|\mathbf{y}\| - \|\hat{\mathbf{y}}\| = 1-1 = 0$. The error vector captures that these point in opposite directions; the magnitude difference misses this entirely.*

5. **In a ResNet with skip connections, the output is $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$. How would you express the residual function $F(\mathbf{x})$ in terms of subtraction?**
   *Answer: $F(\mathbf{x}) = \mathbf{y} - \mathbf{x}$. The residual function is exactly the difference between the output and the input. This is why these are called "residual" networks — the layers learn the residual (difference) rather than the full transformation. If the optimal mapping is close to the identity, the residual is small, making learning easier.*

### Advanced

1. **Prove that $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$ using the definition of vector subtraction and the dot product.**
   *Answer:*
   $$\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v})$$
   $$= \mathbf{u} \cdot \mathbf{u} - \mathbf{u} \cdot \mathbf{v} - \mathbf{v} \cdot \mathbf{u} + \mathbf{v} \cdot \mathbf{v}$$
   $$= \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$$
   
   This identity is the vector form of the Law of Cosines and shows that the squared distance between two vectors depends on their individual magnitudes and the angle between them (through the dot product).

2. **In the triplet loss used for face recognition, the loss is $L = \max(\|f(\mathbf{A}) - f(\mathbf{P})\|^2 - \|f(\mathbf{A}) - f(\mathbf{N})\|^2 + \alpha, 0)$, where $\mathbf{A}$ is an anchor image, $\mathbf{P}$ is a positive (same person), and $\mathbf{N}$ is a negative (different person). Explain how vector subtraction is used to both pull positives together and push negatives apart.**
   *Answer: The term $\|f(\mathbf{A}) - f(\mathbf{P})\|^2$ is the squared distance between anchor and positive embeddings. The subtraction $f(\mathbf{A}) - f(\mathbf{P})$ gives the difference vector, and its norm squared measures how far apart they are. Minimising this term pushes the anchor and positive closer together. The term $\|f(\mathbf{A}) - f(\mathbf{N})\|^2$ measures the distance to the negative. By subtracting this from the positive distance (and adding a margin $\alpha$), the loss encourages that negatives be at least $\alpha$ farther from the anchor than positives. Both terms rely fundamentally on vector subtraction to compute distances.*

3. **Derive the gradient of the MSE loss $L = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$ with respect to the prediction vector $\hat{\mathbf{y}}$. Show that the gradient is $-(\mathbf{y} - \hat{\mathbf{y}})$.**
   *Answer:*
   $$L = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \frac{1}{2}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   
   Taking the partial derivative with respect to $\hat{y}_i$:
   $$\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{2} \cdot 2(y_i - \hat{y}_i) \cdot (-1) = -(y_i - \hat{y}_i)$$
   
   Therefore the gradient vector is:
   $$\nabla_{\hat{\mathbf{y}}} L = \begin{pmatrix} -(y_1 - \hat{y}_1) \\ \vdots \\ -(y_n - \hat{y}_n) \end{pmatrix} = -(\mathbf{y} - \hat{\mathbf{y}})$$
   
   This means the gradient of the MSE loss with respect to the predictions is simply the negative error vector. To reduce the loss, we move the predictions in the direction of the positive error vector (toward the targets), which matches the intuition we build during gradient descent: $\hat{\mathbf{y}}_{\text{new}} = \hat{\mathbf{y}} + \alpha(\mathbf{y} - \hat{\mathbf{y}})$.

## Practice Problems

### Easy - 5 Questions

1. Let $\mathbf{a} = (6, 2)^T$ and $\mathbf{b} = (1, 4)^T$. Compute $\mathbf{a} - \mathbf{b}$.

2. Compute $(5, -3, 2)^T - (2, -1, 4)^T$.

3. If $\mathbf{v} = (-2, 7)^T$, what is $\mathbf{0} - \mathbf{v}$?

4. Point $A$ is at $(0, 3)$ and point $B$ is at $(4, 0)$. Find $\overrightarrow{AB}$.

5. Let $\mathbf{u} = (10, -5)^T$ and $\mathbf{v} = (3, -8)^T$. Compute $\mathbf{u} - \mathbf{v}$ and $\mathbf{v} - \mathbf{u}$. Verify they are negatives.

### Medium - 5 Questions

1. The true labels for 4 data points are $\mathbf{y} = (1, 0, 1, 1)^T$ and predictions are $\hat{\mathbf{y}} = (0.8, 0.3, 0.9, 0.7)^T$. Compute the error vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$.

2. Let $\mathbf{w} = (1.5, -2.0, 0.5)^T$ and $\nabla L = (0.4, -0.1, 0.3)^T$ with $\alpha = 0.2$. Compute $\mathbf{w} - \alpha \nabla L$.

3. Show that $\mathbf{u} - \mathbf{v} - \mathbf{w} = \mathbf{u} - (\mathbf{v} + \mathbf{w})$ for $\mathbf{u} = (5, -1)^T$, $\mathbf{v} = (2, 3)^T$, $\mathbf{w} = (1, -4)^T$. Compute both sides and compare.

4. Given $\mathbf{a} = (3, 7, 1)^T$, $\mathbf{b} = (-1, 2, 5)^T$, find $\mathbf{d} = \mathbf{a} - \mathbf{b}$. Then compute $3\mathbf{d}$.

5. A model's embedding for "cat" is $(0.6, 0.8)^T$, for "kitten" is $(0.5, 0.7)^T$, and for "dog" is $(0.9, 0.3)^T$. Compute $( \text{cat} - \text{kitten} )$ and $( \text{cat} - \text{dog} )$. Which difference is smaller? What does this suggest about the embeddings?

### Hard - 3 Questions

1. Prove that $c(\mathbf{u} - \mathbf{v}) = c\mathbf{u} - c\mathbf{v}$ for any scalar $c$ and vectors $\mathbf{u}, \mathbf{v}$.

2. In contrastive learning, for a batch of $N$ samples, the loss for a positive pair is $\|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2$. Let $f(\mathbf{x}_i) = (0.2, 0.5)^T$ and $f(\mathbf{x}_j) = (0.4, 0.3)^T$. Compute the gradient of the loss with respect to $f(\mathbf{x}_i)$. The gradient of $\|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2$ with respect to $f(\mathbf{x}_i)$ is $2(f(\mathbf{x}_i) - f(\mathbf{x}_j))$. Verify this formula.

3. Using the identity $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$, prove the triangle inequality for vector subtraction: $\|\mathbf{u} - \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$. (Hint: start with $\|\mathbf{u} + (-\mathbf{v})\| \leq \|\mathbf{u}\| + \|-\mathbf{v}\|$ and use $\|\mathbf{-v}\| = \|\mathbf{v}\|$.)

## Solutions

### Easy Solutions

**1.** $\mathbf{a} - \mathbf{b} = (6 - 1, 2 - 4)^T = (5, -2)^T$.

**2.** $(5 - 2, -3 - (-1), 2 - 4)^T = (3, -2, -2)^T$.

**3.** $\mathbf{0} - \mathbf{v} = (0 - (-2), 0 - 7)^T = (2, -7)^T = -\mathbf{v}$.

**4.** $\overrightarrow{AB} = \mathbf{b} - \mathbf{a} = (4 - 0, 0 - 3)^T = (4, -3)^T$.

**5.** $\mathbf{u} - \mathbf{v} = (10 - 3, -5 - (-8))^T = (7, 3)^T$.
$\mathbf{v} - \mathbf{u} = (3 - 10, -8 - (-5))^T = (-7, -3)^T = -\mathbf{u} + \mathbf{v} = -(7, 3)^T = -(\mathbf{u} - \mathbf{v})$. ✓

### Medium Solutions

**1.** $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (1 - 0.8, 0 - 0.3, 1 - 0.9, 1 - 0.7)^T = (0.2, -0.3, 0.1, 0.3)^T$.

**2.** $\alpha \nabla L = 0.2 \times (0.4, -0.1, 0.3)^T = (0.08, -0.02, 0.06)^T$.
$\mathbf{w} - \alpha \nabla L = (1.5 - 0.08, -2.0 - (-0.02), 0.5 - 0.06)^T = (1.42, -1.98, 0.44)^T$.

**3.** Left side: $\mathbf{u} - \mathbf{v} - \mathbf{w}$
First compute $\mathbf{u} - \mathbf{v} = (5 - 2, -1 - 3)^T = (3, -4)^T$.
Then subtract $\mathbf{w}$: $(3 - 1, -4 - (-4))^T = (2, 0)^T$.

Right side: $\mathbf{u} - (\mathbf{v} + \mathbf{w})$
First compute $\mathbf{v} + \mathbf{w} = (2 + 1, 3 + (-4))^T = (3, -1)^T$.
Then subtract: $(5 - 3, -1 - (-1))^T = (2, 0)^T$.

Both sides equal $(2, 0)^T$. ✓ This demonstrates that successive subtractions are equivalent to subtracting the sum.

**4.** $\mathbf{d} = \mathbf{a} - \mathbf{b} = (3 - (-1), 7 - 2, 1 - 5)^T = (4, 5, -4)^T$.
$3\mathbf{d} = 3 \times (4, 5, -4)^T = (12, 15, -12)^T$.

**5.** $\text{cat} - \text{kitten} = (0.6 - 0.5, 0.8 - 0.7)^T = (0.1, 0.1)^T$.
$\text{cat} - \text{dog} = (0.6 - 0.9, 0.8 - 0.3)^T = (-0.3, 0.5)^T$.
The cat-kitten difference has magnitude $\sqrt{0.1^2 + 0.1^2} \approx 0.141$, while cat-dog has magnitude $\sqrt{(-0.3)^2 + 0.5^2} \approx 0.583$. The smaller difference for cat-kitten suggests the embeddings encode semantic similarity — cats and kittens are more related than cats and dogs.

### Hard Solutions

**1.** Starting from the left side:
$$c(\mathbf{u} - \mathbf{v}) = c(\mathbf{u} + (-\mathbf{v})) \quad \text{(definition of subtraction)}$$
$$= c\mathbf{u} + c(-\mathbf{v}) \quad \text{(distributive property)}$$
$$= c\mathbf{u} + (-c\mathbf{v}) \quad \text{(scalar multiplication property: } c(-\mathbf{v}) = -(c\mathbf{v})\text{)}$$
$$= c\mathbf{u} - c\mathbf{v} \quad \text{(definition of subtraction)}$$

Thus $c(\mathbf{u} - \mathbf{v}) = c\mathbf{u} - c\mathbf{v}$. ✓

**2.** Let $\mathbf{z} = f(\mathbf{x}_i) - f(\mathbf{x}_j) = (0.2 - 0.4, 0.5 - 0.3)^T = (-0.2, 0.2)^T$.
The loss is $L = \|\mathbf{z}\|^2 = z_1^2 + z_2^2 = (-0.2)^2 + (0.2)^2 = 0.04 + 0.04 = 0.08$.

Gradient with respect to $f(\mathbf{x}_i)$:
$$\frac{\partial L}{\partial f(\mathbf{x}_i)} = \frac{\partial}{\partial f(\mathbf{x}_i)} \|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2$$

For each component $k$:
$$\frac{\partial L}{\partial f(\mathbf{x}_i)_k} = 2(f(\mathbf{x}_i)_k - f(\mathbf{x}_j)_k)$$

So $\nabla_{f(\mathbf{x}_i)} L = 2(f(\mathbf{x}_i) - f(\mathbf{x}_j)) = 2 \times (-0.2, 0.2)^T = (-0.4, 0.4)^T$.

Verification by direct computation:
$L = (a - b_1)^2 + (a - b_2)^2$ where $a = f(\mathbf{x}_i)_1$, etc.
$\partial L / \partial f(\mathbf{x}_i)_1 = 2(f(\mathbf{x}_i)_1 - f(\mathbf{x}_j)_1) = 2(-0.2) = -0.4$ ✓
$\partial L / \partial f(\mathbf{x}_i)_2 = 2(f(\mathbf{x}_i)_2 - f(\mathbf{x}_j)_2) = 2(0.2) = 0.4$ ✓

**3.** From the triangle inequality for addition, we know $\|\mathbf{a} + \mathbf{b}\| \leq \|\mathbf{a}\| + \|\mathbf{b}\|$.

Let $\mathbf{a} = \mathbf{u}$ and $\mathbf{b} = -\mathbf{v}$. Then:
$$\|\mathbf{u} + (-\mathbf{v})\| \leq \|\mathbf{u}\| + \|-\mathbf{v}\|$$

But $\mathbf{u} + (-\mathbf{v}) = \mathbf{u} - \mathbf{v}$ (definition of subtraction) and $\|-\mathbf{v}\| = \sqrt{(-v_1)^2 + \cdots + (-v_n)^2} = \sqrt{v_1^2 + \cdots + v_n^2} = \|\mathbf{v}\|$.

Therefore:
$$\|\mathbf{u} - \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$$

This is the triangle inequality for subtraction. It tells us that the distance between two vectors is at most the sum of their individual magnitudes. ✓

## Related Concepts

- **Vector Addition (MATH-011):** The inverse relationship — subtraction is addition of the negative.
- **Vector Magnitude (MATH-014):** Used to measure the size of difference vectors.
- **Vector (MATH-002):** The fundamental object being subtracted.
- **Scalar (MATH-001):** The individual components that are subtracted.
- **Dot Product (MATH-016):** Used to compute $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v}$.
- **Cross Product (MATH-017):** Another operation combining two vectors (produces a perpendicular vector in 3D).

## Next Concepts

- **Linear Combinations (MATH-020):** Weighted sums using both addition and subtraction.
- **Vector Projection (MATH-030):** Decomposing one vector into components parallel and perpendicular to another.
- **Distance Metrics (MATH-040):** Different ways to measure the difference between vectors.
- **Gradient Descent (MATH-058):** The optimisation algorithm built on vector subtraction.

## Summary

Vector subtraction is performed component-wise: $\mathbf{u} - \mathbf{v} = (u_1 - v_1, \ldots, u_n - v_n)^T$. It is equivalent to adding the additive inverse: $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$. Geometrically, $\mathbf{u} - \mathbf{v}$ is the vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$, which makes it ideal for computing the displacement between two points ($\overrightarrow{AB} = \mathbf{b} - \mathbf{a}$). Subtraction is not commutative ($\mathbf{u} - \mathbf{v} \neq \mathbf{v} - \mathbf{u}$) but is anti-commutative ($\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$). In AI/ML, vector subtraction is used to compute error vectors, gradient descent updates, word embedding analogies, and contrastive losses.

## Key Takeaways

- Vector subtraction is performed **component-wise**: $(\mathbf{u} - \mathbf{v})_i = u_i - v_i$.
- Subtraction is equivalent to **adding the negative**: $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$.
- Geometrically, $\mathbf{u} - \mathbf{v}$ is the vector from the tip of $\mathbf{v}$ to the tip of $\mathbf{u}$.
- The vector from point $A$ to point $B$ is $\overrightarrow{AB} = \mathbf{b} - \mathbf{a}$.
- Subtraction is **not commutative**: $\mathbf{u} - \mathbf{v} = -(\mathbf{v} - \mathbf{u})$.
- In AI/ML, subtraction is central to loss computation ($\mathbf{y} - \hat{\mathbf{y}}$), gradient descent updates ($\mathbf{w} - \alpha \nabla L$), and embedding analogies.
- The squared distance $\|\mathbf{u} - \mathbf{v}\|^2$ expands to $\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$.
- The triangle inequality for subtraction: $\|\mathbf{u} - \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$.
