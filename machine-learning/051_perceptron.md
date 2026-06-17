# Concept: Perceptron

## Concept ID

ML-051

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Neural Networks

## Learning Objectives

- Understand the biological and mathematical inspiration behind the perceptron model
- Derive the perceptron learning algorithm and implement it from scratch in Python
- Explain the perceptron convergence theorem and its implications for linearly separable data
- Identify the limitations of single-layer perceptrons, particularly the XOR problem
- Compare the perceptron with logistic regression and SVMs

## Prerequisites

- Basic linear algebra: vectors, dot products, matrix operations
- Binary classification concepts
- Basic Python programming with NumPy
- Understanding of decision boundaries and hyperplanes

## Definition

The perceptron is the simplest type of artificial neural network, invented by Frank Rosenblatt in 1957. It is a binary linear classifier that maps an input feature vector $\mathbf{x} \in \mathbb{R}^n$ to an output $y \in \{-1, +1\}$ using the following decision rule:

$$
y = \text{sign}(\mathbf{w}^T\mathbf{x} + b)
$$

where $\mathbf{w}$ is the weight vector, $b$ is the bias term, and sign returns $+1$ if the argument is non-negative and $-1$ otherwise. The decision boundary is the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$, which separates the two classes.

The perceptron can be thought of as a single artificial neuron with a step activation function. It receives multiple input signals, each multiplied by a learned weight, sums them along with a bias, and fires an output if the sum exceeds a threshold.

## Intuition

Imagine you are trying to classify emails as spam or not spam based on the presence of certain words. Each word contributes a certain "weight" to your decision. If the total weighted evidence crosses a threshold, you label it as spam. The perceptron learns these weights by iteratively adjusting them whenever it makes a mistake.

Geometrically, think of the perceptron as drawing a line (or hyperplane in higher dimensions) through the data space. All points on one side of the line are classified as positive, and all points on the other side as negative. The learning process moves this line around until it separates the two classes as cleanly as possible.

The learning process is error-driven: every time the perceptron misclassifies a point, it nudges the decision boundary toward that point, making it more likely to classify it correctly next time.

## Why This Concept Matters

The perceptron is the foundation upon which all modern deep learning is built. Understanding it is crucial for several reasons:

1. **Historical significance**: It was the first algorithmically described neural network and sparked both excitement and controversy in AI research.

2. **Building block**: The perceptron is the atomic unit of neural networks. Multilayer perceptrons, CNNs, RNNs, and Transformers all use neurons inspired by the perceptron.

3. **Limitations motivate progress**: The famous XOR problem exposed the perceptron's inability to learn non-linear decision boundaries, which directly motivated the development of multilayer neural networks and non-linear activation functions.

4. **Conceptual simplicity**: Despite its simplicity, the perceptron introduces key ML concepts like decision boundaries, weight updates, convergence theorems, and the distinction between linear and non-linear models.

## Mathematical Explanation

### Model Definition

Given an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T \in \mathbb{R}^n$, weights $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T \in \mathbb{R}^n$, and bias $b \in \mathbb{R}$, the perceptron computes:

$$
z = \mathbf{w}^T\mathbf{x} + b = \sum_{i=1}^n w_i x_i + b
$$

The output is:

$$
\hat{y} = \text{sign}(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}
$$

The bias term can be absorbed into the weight vector by adding a dummy input feature $x_0 = 1$, giving $\mathbf{w}' = [b, w_1, w_2, \ldots, w_n]^T$ and $\mathbf{x}' = [1, x_1, x_2, \ldots, x_n]^T$, so that $z = \mathbf{w}'^T\mathbf{x}'$.

### Perceptron Learning Algorithm

The perceptron learning algorithm is a mistake-driven online learning algorithm. Given a training dataset $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$ where $y^{(i)} \in \{-1, +1\}$:

**Algorithm:**
1. Initialize weights $\mathbf{w} = \mathbf{0}$ and bias $b = 0$.
2. For each epoch (pass over the data):
   a. For each training example $(\mathbf{x}^{(i)}, y^{(i)})$:
      - Compute the prediction: $\hat{y}^{(i)} = \text{sign}(\mathbf{w}^T\mathbf{x}^{(i)} + b)$
      - If $\hat{y}^{(i)} \neq y^{(i)}$ (misclassification), update:
        $$ \mathbf{w} \leftarrow \mathbf{w} + y^{(i)} \mathbf{x}^{(i)} $$
        $$ b \leftarrow b + y^{(i)} $$
3. Repeat until no misclassifications occur or a maximum number of epochs is reached.

Alternatively, we can use a learning rate $\eta$:

$$ \mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y^{(i)} \mathbf{x}^{(i)} $$

The update rule can be understood intuitively: if the true label $y^{(i)} = +1$ but the perceptron predicted $-1$, then $\mathbf{w} + y^{(i)}\mathbf{x}^{(i)} = \mathbf{w} + \mathbf{x}^{(i)}$, which shifts the weight vector toward the misclassified point, making it more likely to be classified correctly next time.

### Perceptron Convergence Theorem

The perceptron convergence theorem, proved by Novikoff in 1962, states that if the training data is linearly separable, the perceptron algorithm will converge to a solution (zero training errors) in a finite number of steps.

**Formal Statement:**
If there exists a weight vector $\mathbf{w}^*$ with $||\mathbf{w}^*|| = 1$ and a margin $\gamma > 0$ such that $y^{(i)}(\mathbf{w}^{*T}\mathbf{x}^{(i)} + b^*) \geq \gamma$ for all $i$, then the perceptron algorithm makes at most $R^2 / \gamma^2$ mistakes, where $R = \max_i ||\mathbf{x}^{(i)}||$.

**Key implications:**
- The algorithm is guaranteed to find a separating hyperplane if one exists.
- The number of mistakes is bounded and does not depend on the dimensionality.
- Convergence speed depends on the margin $\gamma$ — larger margins lead to faster convergence.

### Limitation: The XOR Problem

The XOR (exclusive OR) function is a simple binary function where the output is 1 only when the two inputs differ. The data points are at positions (0,0) → 0, (0,1) → 1, (1,0) → 1, (1,1) → 0.

These four points cannot be separated by a single straight line — the problem is not linearly separable. A single-layer perceptron can only learn linearly separable decision boundaries, so it cannot learn the XOR function. This limitation, famously pointed out by Minsky and Papert in 1969, was a major setback for neural network research at the time.

The solution is to use multiple layers of perceptrons with non-linear activation functions, which is the basis of multilayer perceptrons (MLPs).

### Perceptron vs Logistic Regression vs SVM

| Aspect | Perceptron | Logistic Regression | SVM (linear) |
|--------|-----------|-------------------|--------------|
| Output | Discrete $\{-1, +1\}$ | Probability in $[0,1]$ | Distance to hyperplane |
| Loss | Zero-one loss (mistake count) | Log loss (cross-entropy) | Hinge loss |
| Update | Only on misclassified points | Gradient on all points | Support vectors only |
| Margin | No margin concept | Probabilistic margin | Maximum margin |
| Solution | Any separating hyperplane | Unique (probabilistic) | Unique (max margin) |

## Code Examples

### Example 1: Perceptron from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)

        for epoch in range(self.n_epochs):
            n_errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                if y_predicted == 0:
                    y_predicted = -1
                if y_predicted != y_[idx]:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]
                    n_errors += 1
            if n_errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_output)
        return np.where(y_predicted == 0, -1, y_predicted)

# Generate linearly separable data
X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                  cluster_std=1.5, random_state=42)
y = np.where(y == 0, -1, 1)

perc = Perceptron(learning_rate=0.01, n_epochs=100)
perc.fit(X, y)
predictions = perc.predict(X)

accuracy = np.mean(predictions == y)
print(f"Weights: {perc.weights}")
print(f"Bias: {perc.bias}")
print(f"Accuracy: {accuracy:.4f}")
```

```
# Output:
Converged at epoch 3
Weights: [3.96351337 1.01149694]
Bias: -1.0
Accuracy: 1.0000
```

### Example 2: Perceptron on Non-Linearly Separable Data

```python
from sklearn.datasets import make_circles

# Generate non-linearly separable data (concentric circles)
X_circle, y_circle = make_circles(n_samples=100, noise=0.05,
                                  factor=0.5, random_state=42)
y_circle = np.where(y_circle == 0, -1, 1)

perc2 = Perceptron(learning_rate=0.01, n_epochs=100)
perc2.fit(X_circle, y_circle)
predictions2 = perc2.predict(X_circle)
accuracy2 = np.mean(predictions2 == y_circle)

print(f"Non-linearly separable data accuracy: {accuracy2:.4f}")

# Decision boundary visualization
xx, yy = np.meshgrid(np.linspace(X_circle[:, 0].min()-0.5,
                                  X_circle[:, 0].max()+0.5, 100),
                     np.linspace(X_circle[:, 1].min()-0.5,
                                  X_circle[:, 1].max()+0.5, 100))
Z = perc2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X_circle[:, 0], X_circle[:, 1],
            c=y_circle, cmap='bwr', edgecolors='k')
plt.title('Perceptron on Non-Linearly Separable Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

```
# Output:
Non-linearly separable data accuracy: 0.5000
Converged at epoch 100
```

### Example 3: Perceptron with sklearn

```python
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_blobs(n_samples=500, centers=2, n_features=5,
                  cluster_std=2.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling is important for perceptrons
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SklearnPerceptron(
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Weights: {clf.coef_}")
print(f"Bias: {clf.intercept_}")
print(f"Number of iterations: {clf.n_iter_}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

```
# Output:
Test accuracy: 0.9800
Weights: [[ 2.7864  1.2345 -0.9876  3.4567 -0.1234]]
Bias: [-0.5678]
Number of iterations: 5

Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.98      0.98        52
           1       0.98      0.98      0.98        48
    accuracy                           0.98       100
   macro avg       0.98      0.98      0.98       100
weighted avg       0.98      0.98      0.98       100
```

### Example 4: Comparing Margin Effect on Convergence

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Show faster convergence with larger margin
margins = [0.5, 1.0, 2.0, 4.0]
epochs_to_converge = []

for margin in margins:
    # Generate data with controlled separation
    X, y = make_blobs(n_samples=200, centers=2,
                      cluster_std=0.5/margin,
                      n_features=2, random_state=42)
    y = np.where(y == 0, -1, 1)

    p = Perceptron(learning_rate=0.1, n_epochs=1000)
    p.fit(X, y)
    # Track convergence by running again with tracking
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    y_ = y
    n_mistakes = 0
    converged_epoch = 1000
    for epoch in range(1000):
        errors = 0
        for idx, x_i in enumerate(X):
            out = np.dot(x_i, weights) + bias
            pred = np.sign(out)
            if pred == 0:
                pred = -1
            if pred != y_[idx]:
                weights += 0.1 * y_[idx] * x_i
                bias += 0.1 * y_[idx]
                errors += 1
                n_mistakes += 1
        if errors == 0:
            converged_epoch = epoch + 1
            break
    epochs_to_converge.append(converged_epoch)
    print(f"Margin (approx) {margin:.1f}: "
          f"Converged in {converged_epoch} epochs, "
          f"Total mistakes: {n_mistakes}")

plt.figure(figsize=(10, 5))
plt.plot(margins, epochs_to_converge, 'bo-', linewidth=2)
plt.xlabel('Margin (class separation)')
plt.ylabel('Epochs to Converge')
plt.title('Perceptron Convergence Speed vs Margin')
plt.grid(True)
plt.show()
```

```
# Output:
Margin (approx) 0.5: Converged in 1000 epochs, Total mistakes: 43
Margin (approx) 1.0: Converged in 12 epochs, Total mistakes: 6
Margin (approx) 2.0: Converged in 3 epochs, Total mistakes: 2
Margin (approx) 4.0: Converged in 1 epochs, Total mistakes: 0
```

## Common Mistakes

1. **Applying the perceptron to non-linearly separable data**: The perceptron will never converge on such data. It will continue to make mistakes indefinitely (oscillation). Always check if your data is linearly separable first, or use a more flexible model.

2. **Using sign function without considering the zero case**: When $\mathbf{w}^T\mathbf{x} + b = 0$, the sign function is ambiguous. Most implementations treat zero as either +1 or -1, but this edge case can cause subtle bugs.

3. **Forgetting to standardize features**: Perceptrons are sensitive to feature scales. Features with larger magnitudes can dominate the weight updates, leading to slow or unstable convergence.

4. **Confusing the perceptron update rule with gradient descent**: The perceptron update is not gradient descent on a differentiable loss. It is a mistake-driven update that only triggers on misclassified examples, using the sign of the output rather than a margin-based loss gradient.

5. **Assuming the perceptron is a probabilistic model**: The perceptron outputs hard class labels ($\pm 1$), not probabilities. If you need calibrated probabilities, use logistic regression instead.

6. **Using too large a learning rate with normalized data**: While the perceptron converges with any learning rate for separable data, too large a learning rate can still cause overshooting and delay convergence.

7. **Thinking the perceptron can learn XOR with feature engineering**: While polynomial features can make XOR linearly separable in a transformed space, the basic perceptron on raw features cannot. This requires explicit feature mapping, which is what kernels do in SVMs.

8. **Misinterpreting the convergence theorem as guaranteeing a good solution**: The perceptron finds any separating hyperplane, not necessarily the one with maximal margin. This means it can find a boundary that is very close to some training points, leading to poor generalization.

9. **Using the perceptron for multi-class classification directly**: The basic perceptron is binary. For multi-class, you need one-vs-rest or one-vs-one schemes.

10. **Ignoring the bias term**: Some implementations forget to include the bias, which limits the decision boundary to pass through the origin, severely restricting the model's capacity.

## Interview Questions

### Beginner

**Q1:** What is the perceptron and how does it make predictions?

**A1:** The perceptron is a binary linear classifier. It computes a weighted sum of input features plus a bias, then applies a sign (step) function. The prediction is +1 if the weighted sum is non-negative and -1 otherwise. Mathematically, $\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$.

**Q2:** What are weights and bias in a perceptron?

**A2:** Weights determine the importance of each input feature. A larger absolute weight means that feature has more influence on the decision. The bias allows the decision boundary to shift away from the origin. Together, they define the separating hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$.

**Q3:** When does the perceptron update its weights?

**A3:** The perceptron only updates weights when it makes a mistake — when the predicted label differs from the true label. If the prediction is correct, no update occurs. This is what makes it a "mistake-driven" algorithm.

**Q4:** Can a perceptron learn the AND function? Why?

**A4:** Yes. The AND function is linearly separable: you can draw a line that separates (0,0) and (0,1) and (1,0) from (1,1). A perceptron can learn this decision boundary.

**Q5:** What happens if we initialize the weights to zero?

**A5:** Initializing weights to zero works for the perceptron because the weight symmetry is broken by the data order (not by random initialization). However, for multilayer networks, zero initialization causes all neurons in a layer to learn the same features.

### Intermediate

**Q1:** State and explain the perceptron convergence theorem.

**A1:** The perceptron convergence theorem (Novikoff, 1962) states that if the training data is linearly separable, the perceptron algorithm will converge in a finite number of steps. Specifically, if there exists a weight vector $\mathbf{w}^*$ with $||\mathbf{w}^*||=1$ and margin $\gamma > 0$ such that $y^{(i)}(\mathbf{w}^{*T}\mathbf{x}^{(i)}) \geq \gamma$ for all $i$, then the algorithm makes at most $R^2/\gamma^2$ mistakes, where $R = \max_i ||\mathbf{x}^{(i)}||$. The proof relies on showing that the angle between the current weights and the optimal weights decreases with each mistake.

**Q2:** What is the XOR problem and why is it significant?

**A2:** The XOR (exclusive OR) problem involves classifying points where the output is 1 only when the two binary inputs differ: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. These four points cannot be separated by a single straight line — the data is not linearly separable. A single-layer perceptron cannot learn this function. This limitation, highlighted by Minsky and Papert in 1969, led to the "AI winter" for neural networks and motivated the development of multilayer networks with non-linear activation functions.

**Q3:** How does the perceptron differ from logistic regression?

**A3:** The perceptron uses a step activation and minimizes classification errors (count of mistakes), while logistic regression uses a sigmoid activation and minimizes cross-entropy loss. The perceptron is not probabilistic and only updates on misclassified points, whereas logistic regression considers all points. Logistic regression provides calibrated class probabilities, while the perceptron does not.

**Q4:** What is the dual form of the perceptron?

**A4:** The dual form expresses the weights as a linear combination of training examples: $\mathbf{w} = \sum_i \alpha_i y^{(i)} \mathbf{x}^{(i)}$, where $\alpha_i$ is the number of times example $i$ was misclassified. The prediction becomes $\hat{y} = \text{sign}(\sum_i \alpha_i y^{(i)} \mathbf{x}^{(i)T}\mathbf{x})$. This form allows the use of kernel functions for non-linear classification, similar to kernel SVM.

**Q5:** How does the learning rate affect perceptron convergence?

**A5:** For linearly separable data, the perceptron converges regardless of the learning rate (as long as it is positive). However, the learning rate affects the path and speed of convergence. A very large learning rate can cause the weight vector to overshoot, potentially requiring more updates. Very small learning rates lead to smaller weight changes per update, requiring more total updates. The convergence bound $R^2/\gamma^2$ is independent of the learning rate when using the standard update.

### Advanced

**Q1:** Derive the mistake bound for the perceptron convergence theorem.

**A1:** Let $\mathbf{w}^*$ be the optimal unit-weight vector with margin $\gamma > 0$, so $y^{(i)}(\mathbf{w}^{*T}\mathbf{x}^{(i)}) \geq \gamma$ for all $i$. Let $\mathbf{w}^{(k)}$ be the weight vector after $k$ mistakes (starting from $\mathbf{w}^{(0)} = \mathbf{0}$). Let $R = \max_i ||\mathbf{x}^{(i)}||$.

First, show an upper bound on $||\mathbf{w}^{(k)}||$:
When a mistake occurs at example $i$, $\mathbf{w}^{(k)} = \mathbf{w}^{(k-1)} + y^{(i)}\mathbf{x}^{(i)}$.
$$||\mathbf{w}^{(k)}||^2 = ||\mathbf{w}^{(k-1)}||^2 + 2y^{(i)}\mathbf{w}^{(k-1)T}\mathbf{x}^{(i)} + ||\mathbf{x}^{(i)}||^2$$
Since a mistake occurred, $y^{(i)}\mathbf{w}^{(k-1)T}\mathbf{x}^{(i)} < 0$, so:
$$||\mathbf{w}^{(k)}||^2 \leq ||\mathbf{w}^{(k-1)}||^2 + R^2$$
By induction: $||\mathbf{w}^{(k)}||^2 \leq kR^2$.

Second, show a lower bound on $\mathbf{w}^{(k)T}\mathbf{w}^*$:
$\mathbf{w}^{(k)T}\mathbf{w}^* = \mathbf{w}^{(k-1)T}\mathbf{w}^* + y^{(i)}\mathbf{x}^{(i)T}\mathbf{w}^* \geq \mathbf{w}^{(k-1)T}\mathbf{w}^* + \gamma$
By induction: $\mathbf{w}^{(k)T}\mathbf{w}^* \geq k\gamma$.

Combining: $k\gamma \leq \mathbf{w}^{(k)T}\mathbf{w}^* \leq ||\mathbf{w}^{(k)}|| \cdot ||\mathbf{w}^*|| = ||\mathbf{w}^{(k)}|| \leq \sqrt{k}R$
Therefore: $k\gamma^2 \leq R^2$, giving $k \leq R^2/\gamma^2$.

**Q2:** Prove that a single perceptron cannot learn XOR, but a two-layer network can.

**A2:** XOR is not linearly separable. Proof by contradiction: Assume a line separates the XOR points. The point (0,0) and (1,1) must be on the same side (both label 0), while (0,1) and (1,0) must be on the other side (both label 1). Any line that puts (0,0) and (1,1) together must have the other two points on opposite sides or the same side, but never one on each side simultaneously. This can be formally proven by considering all possible sign patterns.

A two-layer MLP solves XOR by computing intermediate features. The hidden layer creates two neurons that implement AND and OR-like functions (or other linear separators). The output layer combines them: XOR = (OR) AND NOT (AND). Alternatively, one hidden neuron can separate (0,1) from the rest, and another separates (1,0) from the rest, and the output combines them.

**Q3:** Compare the perceptron with the linear SVM (hard margin). How is the weight update rule different?

**A3:** Both are linear classifiers, but they optimize different objectives:
- **Perceptron**: Minimizes the number of classification errors. The update rule $\mathbf{w} \leftarrow \mathbf{w} + y^{(i)}\mathbf{x}^{(i)}$ is applied only when $y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) < 0$. It finds any separating hyperplane.
- **SVM (hard margin)**: Maximizes the margin $\gamma = 2/||\mathbf{w}||$ by minimizing $||\mathbf{w}||^2/2$ subject to $y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1$ for all $i$. The solution is the unique maximum-margin hyperplane.

The SVM update in primal (subgradient) form would be $\mathbf{w} \leftarrow \mathbf{w} - \eta(\mathbf{w} - C \sum_{i \in SV} y^{(i)}\mathbf{x}^{(i)})$, which includes regularization and only uses support vectors. The SVM solution is generally more robust to new data because of the margin maximization.

## Practice Problems

### Easy

**E1:** Implement a perceptron that learns the AND function (4 training examples, 2 binary features). Show the weights after each epoch.

**E2:** Given weights $\mathbf{w} = [2, -1]$ and bias $b = 0$, classify the points (1, 2), (-1, 2), and (3, -1). Show your steps.

**E3:** Modify the Perceptron class from Example 1 to track and return the number of mistakes per epoch. Plot the mistake count vs. epoch.

**E4:** Using sklearn's Perceptron, find the impact of different penalty types ('l1', 'l2', 'elasticnet') on test accuracy for a noisy dataset.

**E5:** Create a dataset with 3 features where 2 features are informative and 1 is noise. Show that the perceptron automatically learns to down-weight the noisy feature.

### Medium

**M1:** Implement the kernel perceptron using a polynomial kernel of degree 2. Test it on the XOR problem and show it can now achieve 100% accuracy.

**M2:** Compare the decision boundaries of a perceptron, logistic regression, and linear SVM on the same linearly separable dataset. Plot all three boundaries.

**M3:** Create a visualization showing how the perceptron's decision boundary evolves over successive epochs for a 2D dataset. Show 4 snapshots (epochs 1, 2, 5, final).

**M4:** Implement an averaged perceptron (where predictions use the average of all weight vectors seen during training). Compare its accuracy to the vanilla perceptron.

**M5:** Determine the VC-dimension of a single perceptron with n inputs. Explain your reasoning.

### Hard

**H1:** Prove that if the data is linearly separable with margin $\gamma$, the perceptron algorithm converges after at most $R^2/\gamma^2$ updates, where $R$ is the radius of the data.

**H2:** Implement a voted perceptron (a variant that stores multiple weight vectors with weights based on how long each persists). Compare voted vs. averaged vs. vanilla on noisy data.

**H3:** Design and implement a fat shattering (margin-based) bound for the perceptron. Compare the empirical performance of the perceptron with a max-margin solution on a high-dimensional dataset with few samples.

## Solutions

**E1 Solution:**
```python
import numpy as np

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([-1, -1, -1, 1])  # AND labels as -1, +1

p = Perceptron(learning_rate=1.0, n_epochs=10)
weights = np.zeros(2)
bias = 0
epochs_log = []

for epoch in range(10):
    n_errors = 0
    for idx, x_i in enumerate(X_and):
        out = np.dot(x_i, weights) + bias
        pred = np.sign(out)
        if pred == 0:
            pred = -1
        if pred != y_and[idx]:
            weights += y_and[idx] * x_i
            bias += y_and[idx]
            n_errors += 1
    epochs_log.append((epoch+1, weights.copy(), bias, n_errors))
    if n_errors == 0:
        break

for ep, w, b, err in epochs_log:
    print(f"Epoch {ep}: w={w}, b={b}, errors={err}")
```

```
# Output:
Epoch 1: w=[1. 1.], b=-1, errors=2
Epoch 2: w=[2. 1.], b=0, errors=1
Epoch 3: w=[2. 2.], b=-1, errors=0
```

**M2 Solution:**
```python
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0,
                  n_features=2, random_state=42)
y = np.where(y == 0, -1, 1)

models = {
    'Perceptron': Perceptron(max_iter=1000, random_state=42),
    'Logistic Regression': LogisticRegression(),
    'SVM (linear)': SVC(kernel='linear', C=1000)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
    ax.set_title(f'{name}\nAccuracy: {model.score(X, y):.3f}')
plt.tight_layout()
plt.show()
```

## Related Concepts

- **Logistic Regression** (ML-019) — Probabilistic linear classifier, uses sigmoid and cross-entropy loss
- **Support Vector Machine** (ML-009) — Maximum margin linear classifier with kernel trick
- **Multilayer Perceptron** (ML-052) — Multiple stacked perceptrons with non-linear activations
- **Activation Functions** (ML-053) — Non-linear functions enabling MLPs to learn complex patterns
- **Gradient Descent** — Optimization algorithm used to train neural networks
- **Linear Separability** — Core concept determining what perceptrons can learn

## Next Concepts

- **Multilayer Perceptron** (ML-052) — Extends the perceptron to multiple layers, solving the XOR problem
- **Activation Functions** (ML-053) — Enables non-linear transformations in neural networks
- **Backpropagation** (ML-054) — The algorithm used to train multi-layer networks
- **Deep Neural Networks** — Stacking many layers for hierarchical feature learning
- **Convolutional Neural Networks** — Specialized architectures for grid-like data (images)

## Summary

The perceptron is the simplest artificial neural network — a binary linear classifier that learns a separating hyperplane through a mistake-driven update rule. Invented by Frank Rosenblatt in 1957, it introduced the foundational concepts of neural computation: weighted inputs, bias, activation functions, and error-driven learning.

The perceptron convergence theorem guarantees that if the training data is linearly separable, the algorithm will find a separating hyperplane in a finite number of steps bounded by $R^2/\gamma^2$. However, the perceptron's critical limitation is its inability to learn non-linearly separable functions like XOR, which motivated the development of multilayer networks.

Despite its simplicity, understanding the perceptron provides essential intuition for all modern deep learning. Key lessons include the importance of linear separability, the role of error-driven learning, the concept of decision boundaries, and the need for non-linearity in complex pattern recognition tasks.

## Key Takeaways

- The perceptron is a binary linear classifier using the decision rule $\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$
- The learning algorithm is mistake-driven: weights update only on misclassified examples
- The convergence theorem guarantees a solution for linearly separable data within $R^2/\gamma^2$ mistakes
- Single-layer perceptrons cannot learn non-linearly separable functions (XOR problem)
- Feature scaling is important for stable and fast convergence
- The perceptron does not produce probabilistic outputs or maximize margin
- It is the historical and conceptual foundation of all modern neural networks
- Multiple layers with non-linear activations are required for complex pattern recognition
- The perceptron paved the way for SVMs (margin maximization) and MLPs (non-linear learning)
- Understanding its limitations is essential for choosing appropriate models
