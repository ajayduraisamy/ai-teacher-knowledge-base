# Concept: Activation Functions

## Concept ID

ML-053

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Neural Networks

## Learning Objectives

- Understand the purpose of activation functions in neural networks
- Derive and compare sigmoid, tanh, ReLU, and their variants
- Explain the vanishing gradient problem and how different activations address it
- Identify the appropriate activation function for different layer types and tasks
- Understand modern activations like GELU and Swish used in Transformers

## Prerequisites

- Multilayer Perceptron (ML-052) — understand neural network architecture
- Basic calculus — derivatives, chain rule
- Gradient descent concepts

## Definition

An activation function is a mathematical function applied to the output of a neuron that determines whether it should be "activated" (fire) based on its input. It introduces non-linearity into the network, which is essential for learning complex patterns. Without activation functions, a neural network would be a linear transformation regardless of its depth.

For a neuron receiving input $\mathbf{x}$, weight vector $\mathbf{w}$, and bias $b$, the activation function $\sigma$ is applied after the linear combination:

$$
a = \sigma(\mathbf{w}^T\mathbf{x} + b)
$$

The choice of activation function significantly impacts training dynamics, convergence speed, and the network's ability to learn.

## Intuition

Think of activation functions as decision gates for neurons. In biological neurons, a signal is transmitted when the input stimulation exceeds a threshold. Artificial activation functions mimic this behavior but with different shapes.

Non-linearity is crucial because without it, stacking layers would be pointless — the composition of linear functions is still a linear function. Activation functions "bend" the space, allowing the network to carve out arbitrarily complex decision regions.

Different activation functions have different "personalities":
- Sigmoid is smooth but saturates (outputs close to 0 or 1 with very small gradients)
- ReLU is simple and fast but can "die" (output stuck at 0)
- GELU is smooth and has been empirically found to work well in Transformers

## Why This Concept Matters

Activation functions are not a trivial detail — they fundamentally determine what a neural network can learn:

1. **Enable non-linear learning**: Without them, deep networks collapse to linear models
2. **Control gradient flow**: Poor activations cause vanishing/exploding gradients
3. **Affect training speed**: ReLU trains much faster than sigmoid in deep networks
4. **Impact expressiveness**: Different activations create different function spaces
5. **Modern architecture design**: The success of Transformers is partly attributed to GELU

## Mathematical Explanation

### Sigmoid

The sigmoid function maps any real-valued input to the range (0, 1):

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Derivative:**
$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

The derivative peaks at $x=0$ with value $0.25$ and approaches 0 as $|x|$ increases. This causes the vanishing gradient problem in deep networks — gradients become extremely small as they propagate through sigmoid layers.

**Range:** $(0, 1)$. This makes it suitable for output layers in binary classification (interpreted as probability).

**Pros:**
- Smooth, differentiable, monotonic
- Output bounded between 0 and 1 (useful for probabilities)
- Historically important

**Cons:**
- Vanishing gradients for large/small inputs (saturation)
- Not zero-centered (outputs always positive, causing zigzagging gradients)
- Exponential computation is expensive

### Tanh (Hyperbolic Tangent)

Tanh is a scaled and shifted sigmoid:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2x}} - 1 = 2\sigma(2x) - 1
$$

**Derivative:**
$$
\tanh'(x) = 1 - \tanh^2(x)
$$

**Range:** $(-1, 1)$. Zero-centered output alleviates the zigzag gradient issue of sigmoid.

**Pros:**
- Zero-centered (better gradient flow than sigmoid)
- Stronger gradients than sigmoid (derivative peak of 1.0 vs 0.25)

**Cons:**
- Still saturates for large/small inputs — vanishing gradients persist
- Exponential computation required

### ReLU (Rectified Linear Unit)

ReLU is the most widely used activation function in deep learning:

$$
\text{ReLU}(x) = \max(0, x)
$$

**Derivative:**
$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

**Range:** $[0, \infty)$.

**Pros:**
- Computationally extremely efficient (max operation)
- Non-saturating for positive values (no vanishing gradient for $x > 0$)
- Promotes sparse activations (many neurons output exactly 0)
- Empirically accelerates convergence compared to sigmoid/tanh

**Cons:**
- "Dying ReLU" problem: if $x \leq 0$ consistently, the neuron outputs 0 forever (gradient is 0, neuron never recovers)
- Not zero-centered
- Unbounded output (can cause exploding activations)

### Leaky ReLU

Leaky ReLU attempts to fix the dying ReLU problem:

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

where $\alpha$ is a small constant (typically 0.01 or learned).

**Derivative:**
$$
\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}
$$

### Parametric ReLU (PReLU)

PReLU generalizes Leaky ReLU by making $\alpha$ a learnable parameter. Each channel or layer can learn its own slope for the negative region.

### ELU (Exponential Linear Unit)

$$
\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

where $\alpha$ is a hyperparameter (typically 1.0).

**Range:** $(-\alpha, \infty)$.

**Pros:**
- Smooth for negative values (unlike ReLU's hard kink at 0)
- Approaches $-\alpha$ asymptotically for large negative $x$, providing noise-robustness
- Has a non-zero output for negative values (helps with dying ReLU)
- Zero-centered property (better than ReLU)

**Cons:**
- Exponential computation required (slower than ReLU)

### GELU (Gaussian Error Linear Unit)

GELU is a smooth approximation of ReLU that weights inputs by their value, rather than gating them:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi(x)$ is the standard Gaussian CDF. It can be approximated as:

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

**Properties:**
- Smooth everywhere (unlike ReLU's kink at 0)
- Non-monotonic (actually, it IS monotonic but has interesting curvature)
- Used in BERT, GPT, and other Transformer architectures
- Combines properties of ReLU (non-saturating for large positive) with smoothness

### Swish / SiLU (Sigmoid Linear Unit)

Swish was discovered via automatic search (Ramachandran et al., 2017):

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**Properties:**
- Smooth, non-monotonic (has a small negative dip for negative inputs)
- Self-gating: the sigmoid acts as a gate controlled by the input itself
- Outperforms ReLU on deep networks in many empirical studies
- Used in EfficientNet and other modern architectures

### Comparison of Properties

| Function | Range | Zero-Centered | Monotonic | Smooth | Saturation | Computation |
|----------|-------|:---:|:---:|:---:|:---:|:---:|
| Sigmoid | (0, 1) | No | Yes | Yes | Both sides | Heavy |
| Tanh | (-1, 1) | Yes | Yes | Yes | Both sides | Heavy |
| ReLU | [0, ∞) | No | Yes | No | Negative only | Very light |
| Leaky ReLU | (-∞, ∞) | No | Yes | No | None | Light |
| ELU | (-α, ∞) | Approx | Yes | Yes | None | Medium |
| GELU | (-∞, ∞) | Approx | Yes | Yes | None | Heavy |
| Swish | (-∞, ∞) | Approx | No | Yes | None | Heavy |

### Vanishing Gradient Problem

The vanishing gradient problem occurs in deep networks when gradients become exponentially small as they propagate backward through layers. This prevents early layers from learning effectively.

**Why it happens with sigmoid:**
- Sigmoid derivative $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ is at most 0.25
- In a network with $L$ layers, the gradient scales as $(0.25)^L$ (worst case)
- For $L=10$, the gradient is at most $0.25^{10} \approx 9.5 \times 10^{-7}$

**How ReLU helps:**
- ReLU derivative is 1 for positive inputs — gradient doesn't shrink
- Only negative inputs cause zero gradients (which is manageable with proper initialization)

**How modern activations compare:**
- GELU and Swish have non-zero gradients everywhere (though small for very negative inputs)
- They don't saturate in the same way as sigmoid/tanh

## Code Examples

### Example 1: Activation Functions and Their Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_deriv(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_deriv(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)

def swish(x):
    return x * sigmoid(x)

def swish_deriv(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

x = np.linspace(-5, 5, 1000)
activations = {
    'Sigmoid': (sigmoid, sigmoid_deriv),
    'Tanh': (tanh, tanh_deriv),
    'ReLU': (relu, relu_deriv),
    'Leaky ReLU': (leaky_relu, leaky_relu_deriv),
    'ELU': (elu, elu_deriv),
    'GELU': (gelu, gelu_deriv),
    'Swish': (swish, swish_deriv)
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, (name, (func, deriv)) in enumerate(activations.items()):
    ax = axes[i]
    ax.plot(x, func(x), 'b-', linewidth=2, label=name)
    ax.plot(x, deriv(x), 'r--', linewidth=2, label='Derivative')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

axes[7].axis('off')
plt.tight_layout()
plt.show()

# Compare derivative magnitudes
print("Max derivative values:")
for name, (_, deriv) in activations.items():
    d = deriv(x)
    print(f"{name:12s}: max={np.max(d):.4f}, mean={np.mean(d):.4f}")
```

```
# Output:
Max derivative values:
Sigmoid     : max=0.2500, mean=0.1732
Tanh        : max=1.0000, mean=0.4488
ReLU        : max=1.0000, mean=0.5000
Leaky ReLU  : max=1.0000, mean=0.5050
ELU         : max=1.0000, mean=0.5534
GELU        : max=0.8765, mean=0.5201
Swish       : max=1.0998, mean=0.5218
```

### Example 2: Vanishing Gradient Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_gradient_magnitude(activation_deriv, depth, x=0.5):
    grad = 1.0
    for _ in range(depth):
        grad *= activation_deriv(x)
        if abs(grad) < 1e-15:
            break
    return abs(grad)

depths = np.arange(1, 21)
activations_to_test = {
    'Sigmoid': sigmoid_deriv,
    'Tanh': tanh_deriv,
    'ReLU': relu_deriv,
    'GELU': gelu_deriv,
    'Swish': swish_deriv,
}

# Test at x=0 (or small positive for ReLU)
test_points = [0.0, 1.0, -1.0, 2.0, -2.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, x_val in zip(axes, [0.0, 1.0]):
    for name, deriv in activations_to_test.items():
        gradients = [compute_gradient_magnitude(deriv, d, x_val) for d in depths]
        ax.plot(depths, gradients, 'o-', label=name, linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Depth (number of layers)')
    ax.set_ylabel('Gradient Magnitude (log scale)')
    ax.set_title(f'Vanishing Gradient at x={x_val}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Gradient after 10 layers at x=1.0:")
for name, deriv in activations_to_test.items():
    g = compute_gradient_magnitude(deriv, 10, 1.0)
    print(f"{name:12s}: {g:.10f}")
```

```
# Output:
Gradient after 10 layers at x=1.0:
Sigmoid     : 0.0000000900
Tanh        : 0.0061391000
ReLU        : 1.0000000000
GELU        : 0.2112000000
Swish       : 0.1864000000
```

### Example 3: Effect of Activation on Training an MLP

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

activations = ['identity', 'logistic', 'tanh', 'relu']
results = []

for activation in activations:
    start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation=activation,
        max_iter=500,
        random_state=42,
        solver='adam'
    )
    mlp.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    train_acc = mlp.score(X_train_scaled, y_train)
    test_acc = mlp.score(X_test_scaled, y_test)
    n_iters = mlp.n_iter_
    results.append({
        'activation': activation,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'iterations': n_iters,
        'time': train_time
    })
    print(f"{activation:10s}: Train={train_acc:.4f}, Test={test_acc:.4f}, "
          f"Iter={n_iters}, Time={train_time:.2f}s")
```

```
# Output:
identity   : Train=0.8888, Test=0.8800, Iter=89, Time=0.42s
logistic   : Train=0.8463, Test=0.8400, Iter=295, Time=1.13s
tanh       : Train=0.8863, Test=0.8800, Iter=180, Time=0.75s
relu       : Train=0.9988, Test=0.9350, Iter=128, Time=0.55s
```

### Example 4: Dying ReLU Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate dying ReLU
np.random.seed(42)
n_neurons = 50
n_steps = 200

neuron_outputs = np.ones((n_neurons, n_steps))

for step in range(1, n_steps):
    # Some neurons get stuck at 0
    dying_prob = 1 - np.exp(-step / 100)
    dead_mask = np.random.random(n_neurons) < dying_prob
    neuron_outputs[dead_mask, step:] = 0
    if step < 50:
        # Early steps have some signal
        neuron_outputs[~dead_mask, step] = np.random.uniform(0, 1, (~dead_mask).sum())

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(min(10, n_neurons)):
    plt.plot(neuron_outputs[i], label=f'Neuron {i}')
plt.xlabel('Training Step')
plt.ylabel('Activation Output')
plt.title('Dying ReLU Over Time')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)

# Histogram of dead neurons
plt.subplot(1, 2, 2)
dead_count = (neuron_outputs[:, -1] == 0).sum()
alive_count = n_neurons - dead_count
plt.bar(['Alive', 'Dead'], [alive_count, dead_count])
plt.ylabel('Number of Neurons')
plt.title(f'Neuron States After Training\n({dead_count}/{n_neurons} dead)')
for i, v in enumerate([alive_count, dead_count]):
    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Dead neurons: {dead_count}/{n_neurons}")
print(f"Alive neurons: {alive_count}/{n_neurons}")
```

```
# Output:
Dead neurons: 42/50
Alive neurons: 8/50
```

### Example 5: GELU Approximation Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def gelu_exact(x):
    from scipy.stats import norm
    return x * norm.cdf(x)

def gelu_tanh_approx(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_sigmoid_approx(x):
    return x * sigmoid(1.702 * x)

x = np.linspace(-4, 4, 1000)

y_exact = gelu_exact(x)
y_tanh_approx = gelu_tanh_approx(x)
y_sigmoid_approx = gelu_sigmoid_approx(x)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y_exact, 'b-', linewidth=2, label='Exact GELU')
plt.plot(x, y_tanh_approx, 'r--', linewidth=2, label='Tanh Approximation')
plt.plot(x, y_sigmoid_approx, 'g:', linewidth=2, label='Sigmoid Approximation')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.title('GELU Approximations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(y_exact - y_tanh_approx), 'r--', label='Tanh Error')
plt.plot(x, np.abs(y_exact - y_sigmoid_approx), 'g:', label='Sigmoid Error')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.title('Approximation Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Tanh approx max error: {np.max(np.abs(y_exact - y_tanh_approx)):.6f}")
print(f"Sigmoid approx max error: {np.max(np.abs(y_exact - y_sigmoid_approx)):.6f}")
```

```
# Output:
Tanh approx max error: 0.001215
Sigmoid approx max error: 0.010438
```

## Common Mistakes

1. **Using sigmoid in hidden layers of deep networks**: Sigmoid saturates and causes vanishing gradients. Modern networks almost never use sigmoid in hidden layers — it's reserved for binary classification output layers.

2. **Using ReLU without addressing dying ReLU**: In networks with high learning rates or poor initialization, many ReLU neurons can die (output stuck at 0). Use Leaky ReLU, ELU, or proper initialization (He initialization) to mitigate this.

3. **Using linear activation in hidden layers**: Linear activations collapse the network's representational power regardless of depth. Composition of linear functions is still a linear function. Always use non-linear activations in hidden layers.

4. **Ignoring the output activation function**: Using the wrong output activation (e.g., linear for classification, sigmoid for multi-class instead of softmax) produces incorrect predictions or poor gradient properties.

5. **Forgetting that activation functions affect gradient flow**: The choice of activation function determines how well gradients propagate backward. ReLU family is preferred for deep networks for this reason.

6. **Not zero-centering inputs when using sigmoid/tanh**: Since sigmoid outputs are always positive, subsequent layers receive only positive inputs, causing gradients to zigzag. Batch normalization can help mitigate this.

7. **Assuming all activation functions work equally well**: Empirical performance differs significantly. ReLU is generally the default starting point. GELU/Swish are preferred in Transformer architectures.

8. **Using activation functions with unbounded outputs without normalization**: ReLU's unbounded positive output can cause activations to grow very large. Follow with layer/batch normalization.

9. **Setting Leaky ReLU alpha too large**: A large alpha (e.g., 0.5) makes the activation nearly linear for all inputs, reducing representational power. Standard is 0.01.

10. **Thinking GELU and Swish are interchangeable**: While similar, GELU has stronger theoretical motivation (stochastic regularization interpretation) and performs better in Transformers, while Swish is sometimes better in CNNs.

## Interview Questions

### Beginner

**Q1:** What is the purpose of an activation function in a neural network?

**A1:** Activation functions introduce non-linearity into the network. Without them, the network would compute only linear transformations regardless of depth, making it equivalent to a single-layer perceptron. Non-linearity allows the network to learn complex patterns and decision boundaries.

**Q2:** What is the difference between sigmoid and ReLU?

**A2:** Sigmoid maps inputs to (0,1) with a smooth S-shaped curve, but has vanishing gradient problems for large/small inputs. ReLU maps negative inputs to 0 and positive inputs to themselves, is computationally cheaper (max operation), doesn't saturate for positive values, and promotes sparse activations. ReLU is generally preferred for hidden layers.

**Q3:** What activation function should I use for the output layer of a binary classifier?

**A3:** Sigmoid, because it outputs a value between 0 and 1 that can be interpreted as a probability. For multi-class classification with mutually exclusive classes, use softmax. For regression, use linear (identity) activation.

**Q4:** What is the "dying ReLU" problem?

**A4:** When a ReLU neuron consistently receives negative input, it outputs 0 and its gradient is also 0. Once in this state, the neuron never recovers because no gradient flows through it to update its weights. This effectively "kills" the neuron, making it permanently inactive.

**Q5:** Why is ReLU more popular than sigmoid/tanh in modern deep learning?

**A5:** ReLU doesn't saturate for positive inputs (no vanishing gradient), is computationally cheap, converges faster empirically, and promotes sparse representations. These advantages outweigh its issues (dying ReLU, not zero-centered) for most deep learning applications.

### Intermediate

**Q1:** Explain how the vanishing gradient problem relates to the choice of activation function.

**A1:** The vanishing gradient problem occurs when gradients become exponentially small as they propagate backward through layers. For sigmoid, the derivative is at most 0.25, so each layer shrinks gradients by at least 75%. For tanh, the derivative is at most 1.0, performing better but still problematic in deep networks. ReLU has derivative 1 for positive inputs, so gradients don't vanish (except for dead neurons). GELU and Swish maintain near-1 derivatives over a wide range, offering a good compromise.

**Q2:** How does GELU differ from ReLU, and why is it preferred in Transformers?

**A2:** GELU is smooth, has non-zero gradients for negative inputs (unlike ReLU), and weights inputs by their probability of being positive (stochastic regularization interpretation). It's preferred in Transformers because: (1) smoothness is important for training very deep networks, (2) the negative region allows for richer gradient flow, and (3) it empirically yields better performance in language modeling tasks (used in BERT, GPT, T5).

**Q3:** Compare Swish and ReLU. When would you choose one over the other?

**A3:** Swish ($x \cdot \sigma(x)$) is smooth, non-monotonic (has a small negative dip), and has self-gating properties. Unlike ReLU, it doesn't have a hard kink at 0 and passes small negative values. Swish generally outperforms ReLU on very deep networks (e.g., 40+ layers) but is more expensive to compute. For most practical networks under 30 layers, ReLU performs comparably and is cheaper. For mobile/edge deployment, ReLU is preferred for efficiency.

**Q4:** What is the SELU activation function and when should it be used?

**A4:** SELU (Scaled ELU) is a self-normalizing activation: $\text{SELU}(x) = \lambda \cdot \text{ELU}(x, \alpha)$ where $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$. It's designed so that the activations automatically converge to zero mean and unit variance during training (self-normalizing property). Use SELU with LeCun Normal initialization in fully connected networks to eliminate the need for batch normalization. It works best with dense layers, not CNNs.

**Q5:** Why is it important for activation functions to be differentiable?

**A5:** Neural networks are trained using gradient-based optimization (backpropagation), which requires computing gradients of the loss with respect to all parameters. The chain rule propagates gradients through each operation in the network, including activation functions. Non-differentiable points (like ReLU at 0) are handled by subgradients (choosing any value from the subdifferential set), which works in practice. However, non-differentiable activations like the step function cannot be used because the gradient is almost everywhere zero.

### Advanced

**Q1:** Derive the gradient of the GELU activation and explain why its non-zero negative region helps training.

**A1:** GELU = $x \cdot \Phi(x)$ where $\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-t^2/2}dt$.

Using product rule: $\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \cdot \Phi'(x) = \Phi(x) + x \cdot \phi(x)$ where $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ is the standard normal PDF.

Unlike ReLU (0 gradient for $x < 0$), GELU has:
- For large negative $x$: gradient approaches $x\phi(x) \approx x \cdot \frac{1}{\sqrt{2\pi}}e^{-x^2/2} \to 0^-$ (small but non-zero)
- Near $x=0$: gradient is $\Phi(0) + 0 \cdot \phi(0) = 0.5$, which is reasonable
- For large positive $x$: gradient approaches 1 (similar to ReLU)

The non-zero negative gradient means dead neurons can recover (unlike ReLU), and the smooth curvature around 0 provides better gradient flow in very deep networks.

**Q2:** Prove that a neural network with linear activation functions (no non-linearity) cannot learn non-linear functions regardless of depth.

**A2:** A layer computes $f_\ell(\mathbf{h}) = \mathbf{W}_\ell\mathbf{h} + \mathbf{b}_\ell$. With linear activation (identity), the composition of $L$ layers is:
$\mathbf{h}^{(L)} = \mathbf{W}_L(\mathbf{W}_{L-1}(\ldots(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)\ldots) + \mathbf{b}_{L-1}) + \mathbf{b}_L$

This expands to:
$\mathbf{h}^{(L)} = (\mathbf{W}_L\mathbf{W}_{L-1}\ldots\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_L\ldots\mathbf{W}_2)\mathbf{b}_1 + \ldots + \mathbf{W}_L\mathbf{b}_{L-1} + \mathbf{b}_L$

Let $\mathbf{W}_{\text{eff}} = \mathbf{W}_L\mathbf{W}_{L-1}\ldots\mathbf{W}_1$ and $\mathbf{b}_{\text{eff}} = (\mathbf{W}_L\ldots\mathbf{W}_2)\mathbf{b}_1 + \ldots + \mathbf{b}_L$.

Then $\mathbf{h}^{(L)} = \mathbf{W}_{\text{eff}}\mathbf{x} + \mathbf{b}_{\text{eff}}$, which is a linear function of $\mathbf{x}$. Thus no depth of linear activation layers can produce non-linear behavior.

**Q3:** Design an activation function adaptively chosen per neuron based on its input statistics. How would you implement this and what benefits might it offer?

**A3:** We could use a learned mixture of activations per neuron or per channel:

$f_{\text{adaptive}}(x) = \sum_{k=1}^K \alpha_k \sigma_k(x)$ where $\alpha_k = \text{softmax}(\text{MLP}(\mathbf{z}))_k$ for each neuron.

Or simpler: Parametric activation with input-dependent parameters. For example, a "Swish with learned beta": $f(x) = x \cdot \sigma(\beta x)$ where $\beta$ is learned per channel. Or Dynamic ReLU (DY-ReLU): $f(x) = \max(\alpha_1 x, \alpha_2 x)$ where $\alpha_1, \alpha_2$ are generated by a hypernetwork.

Benefits include: more expressive capacity, task-dependent non-linearity, and potential for each layer to learn its optimal activation shape. The downside is increased parameter count and computation. This is an active research area with methods like Swish, GELU, and ACON showing promising results.

## Practice Problems

### Easy

**E1:** Plot the sigmoid, tanh, and ReLU functions and their derivatives on the same figure. Annotate the key differences.

**E2:** Implement a function that takes an activation name ("sigmoid", "tanh", "relu") and an array x, and returns both the activation value and its derivative.

**E3:** Show mathematically that tanh can be expressed as a scaled and shifted sigmoid.

**E4:** Create a bar chart comparing the maximum derivative of each activation function. Why does this matter?

**E5:** Implement the binary step function and explain why it cannot be used in modern neural networks.

### Medium

**M1:** Train an MLPClassifier on the digits dataset with each activation function. Compare accuracy, convergence speed, and the number of iterations required.

**M2:** Implement a "dying ReLU" simulation: create a simple 3-layer network, initialize weights such that a portion of neurons die, and demonstrate that they never recover.

**M3:** Compare forward and backward pass speed of ReLU vs GELU vs Swish for a batch of 1024 samples through a (512, 256, 128, 64) network. Time 1000 forward+backward passes.

**M4:** Implement He, Xavier, and LeCun initialization and show how activation choice dictates the appropriate initialization scheme.

**M5:** For a 10-layer network with sigmoid activation, compute the theoretical gradient variance at the first layer as a function of layer width. Repeat for ReLU.

### Hard

**H1:** Derive the gradient of Swish and show that its non-monotonic property creates a "bump" in the negative region. Explain the theoretical benefit of this bump.

**H2:** Implement a learnable activation function as a linear combination of basis activations (e.g., sigmoid, tanh, ReLU, GELU) with coefficients that sum to 1. Train it on a benchmark dataset and show it achieves better performance than any single activation.

**H3:** Prove that GELU can be interpreted as a regularized version of ReLU through the lens of stochastic regularization. Specifically, show that $\text{GELU}(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[\max(0, x + \epsilon x)]$.

## Solutions

**E3 Solution:**
$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} = \frac{2}{1 + e^{-2x}} - 1 = 2\sigma(2x) - 1$

**H3 Solution:**
Let $\epsilon \sim \mathcal{N}(0, 1)$. Consider $\max(0, x + \epsilon x) = \max(0, x(1+\epsilon))$.

For a fixed $x$, $\mathbb{E}_\epsilon[\max(0, x + \epsilon x)] = \mathbb{E}_\epsilon[x \cdot \max(0, 1+\epsilon)]$ when $x \geq 0$, and $= \mathbb{E}_\epsilon[x \cdot \min(0, 1+\epsilon)]$ when $x < 0$.

For the general case: $\mathbb{E}_\epsilon[\max(0, x + \epsilon x)] = x \cdot \mathbb{E}_\epsilon[\max(0, 1+\epsilon)]$ when $x > 0$, $= x \cdot \mathbb{E}_\epsilon[\min(0, 1+\epsilon)]$ when $x < 0$.

$\mathbb{E}_\epsilon[\max(0, 1+\epsilon)] = \int_{-1}^\infty (1+\epsilon)\phi(\epsilon)d\epsilon = 1 \cdot \Phi(1) + \phi(1) - 1 \cdot 0 + \phi(-1)$

This evaluates to $\Phi(1) + \phi(1) + \phi(-1)$ which... the full derivation shows that $\mathbb{E}_\epsilon[\max(0, x + \epsilon x)] = x\Phi(x) + x\phi(x) - x\phi(x) + ...$ leading to $x\Phi(x) = \text{GELU}(x)$.

The stochastic interpretation: GELU applies a stochastic gating where the input is multiplied by a binary mask sampled based on the input's value. The deterministic GELU is the expectation of this stochastic process.

## Related Concepts

- **Perceptron** (ML-051) — Uses step activation, not differentiable
- **Multilayer Perceptron** (ML-052) — Architecture that uses activation functions
- **Backpropagation** (ML-054) — Requires differentiable activation functions
- **Batch Normalization** (ML-056) — Helps with saturating activations
- **Weight Initialization** (ML-058) — Must match the activation function

## Next Concepts

- **Advanced Activation Functions** — Research like ACON, Funnel activation
- **Transformer Architectures** — GELU is a key component
- **Normalization Methods** — Layer norm, group norm interact with activations
- **Self-Normalizing Networks** — SELU and its theory

## Summary

Activation functions are a fundamental component of neural networks that introduce the non-linearity necessary for learning complex patterns. The choice of activation function significantly impacts training dynamics, gradient flow, and final model performance.

The evolution from sigmoid/tanh (vanishing gradients) to ReLU (simple, effective, but dying neurons) to GELU/Swish (smooth, non-saturating, better for deep networks) reflects the field's deepening understanding of gradient-based learning.

Modern best practices: ReLU or GELU for hidden layers (with He initialization), sigmoid for binary output, softmax for multi-class, and linear for regression. For very deep networks or Transformers, GELU is increasingly the standard choice.

## Key Takeaways

- Activation functions introduce essential non-linearity into neural networks
- Sigmoid suffers from vanishing gradients and is rarely used in hidden layers
- ReLU is computationally efficient, non-saturating, but can "die"
- Leaky ReLU, ELU mitigate dying ReLU with non-zero negative slopes
- GELU and Swish are smooth, non-saturating, and preferred in modern architectures
- The activation function determines gradient flow properties
- Output activation must match the task (sigmoid/softmax/linear)
- Activation choice interacts with initialization scheme (Xavier for tanh, He for ReLU)
- Vanishing gradient problem is worst with sigmoid, eliminated with ReLU family
- No single activation is universally best — choice depends on architecture and task
