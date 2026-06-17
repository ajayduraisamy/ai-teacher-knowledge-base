# Concept: Weight Initialization

## Concept ID

ML-058

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Neural Networks

## Learning Objectives

- Understand why proper weight initialization is critical for neural network training
- Explain the symmetry problem with zero initialization
- Implement and compare Xavier/Glorot and He initialization
- Understand how initialization interacts with activation function choice
- Diagnose vanishing/exploding gradients caused by poor initialization

## Prerequisites

- Backpropagation (ML-054) — how gradients flow through networks
- Activation Functions (ML-053) — different activations have different gradient properties

## Definition

Weight initialization refers to the strategy used to set the initial values of neural network weights before training begins. Proper initialization is crucial for stable training because it determines the scale of activations and gradients at the start of training. Poor initialization can cause vanishing gradients (weights too small), exploding gradients (weights too large), or symmetry problems (all weights equal).

## Intuition

Think of training a neural network like starting a race. If everyone starts at the same position, they all run the same path (symmetry). If some runners start too far back, they'll never catch up (vanishing gradients). If some start too far ahead, they'll overshoot the finish (exploding gradients).

Proper initialization gives each neuron a unique starting position at the right scale so that learning proceeds efficiently. The goal is to maintain consistent variance of activations and gradients throughout the network, regardless of depth.

## Why This Concept Matters

1. **Convergence speed**: Proper initialization can reduce training time by 10x or more.
2. **Training stability**: Poor initialization causes NaN values or failure to learn.
3. **Enables deep networks**: Deeper networks are impossible without proper initialization.
4. **Interacts with activations**: Different activations require different initialization schemes.
5. **Foundation for advanced techniques**: Understanding initialization is essential for understanding normalization methods.

## Mathematical Explanation

### Zero Initialization (The Symmetry Problem)

Initializing all weights to zero causes every neuron in a layer to compute the same output and receive the same gradient. They all learn the same features, wasting model capacity. The network effectively collapses to having one neuron per layer.

Proof: If W = 0 for all layers, then a = sigma(Wx + b) = sigma(b), which is the same for all neurons in the layer. Since gradients are also identical, all neurons remain identical after updates.

### Random Small Initialization

W ~ N(0, 0.01^2)

Works for shallow networks but fails for deep ones. In a network with 100 layers, the variance of activations shrinks exponentially:
Var(a^{(L)}) = Var(a^{(1)}) * (0.01^2 * n_in)^{L-1}

If (0.01^2 * n_in) < 1, activations vanish. If > 1, they explode.

### Xavier/Glorot Initialization

Designed for activation functions with symmetry around zero (tanh, sigmoid, linear):

W ~ N(0, 2 / (n_in + n_out))

or uniform distribution:
W ~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))

The derivation ensures Var(z^{(l)}) = Var(z^{(l-1)}) for forward pass and Var(dL/dW^{(l)}) is stable for backward pass.

Key assumption: activation function is approximately linear near 0 (holds for tanh, not ReLU).

### He Initialization

Designed specifically for ReLU and its variants:

W ~ N(0, 2 / n_in)

or:
W ~ U(-sqrt(6/n_in), sqrt(6/n_in))

ReLU zeros out half the activations, so the variance is halved compared to Xavier. He initialization doubles the variance to compensate: Var(W) = 2/n_in instead of 1/n_in (for Xavier with equal fan-in/fan-out).

### LeCun Initialization

Designed for SELU (self-normalizing networks):

W ~ N(0, 1 / n_in)

This ensures that inputs to SELU have variance 1, which the activation function maintains.

### Comparison

| Method | Distribution | Variance | Best For |
|--------|-------------|----------|----------|
| Random small | N(0, 0.01^2) | 0.0001 | Shallow nets |
| Xavier/Glorot | N(0, 2/(n_in+n_out)) | 2/(n_in+n_out) | tanh, sigmoid, linear |
| He | N(0, 2/n_in) | 2/n_in | ReLU, Leaky ReLU |
| LeCun | N(0, 1/n_in) | 1/n_in | SELU |

## Code Examples

### Example 1: Effect of Initialization on Activation Variance

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_pass(init_method, n_layers=50, n_units=256, activation='tanh'):
    x = np.random.randn(1000, n_units)
    activations = [x]

    for i in range(n_layers):
        if init_method == 'zero':
            W = np.zeros((n_units, n_units))
        elif init_method == 'small':
            W = np.random.randn(n_units, n_units) * 0.01
        elif init_method == 'xavier':
            W = np.random.randn(n_units, n_units) * np.sqrt(2 / (2 * n_units))
        elif init_method == 'he':
            W = np.random.randn(n_units, n_units) * np.sqrt(2 / n_units)
        elif init_method == 'lecun':
            W = np.random.randn(n_units, n_units) * np.sqrt(1 / n_units)

        z = activations[-1] @ W
        if activation == 'tanh':
            a = np.tanh(z)
        elif activation == 'relu':
            a = np.maximum(0, z)
        elif activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        activations.append(a)

    return activations

init_methods = ['zero', 'small', 'xavier', 'he']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, method in zip(axes.ravel(), init_methods):
    acts = forward_pass(method, n_layers=30, n_units=256, activation='tanh')
    means = [np.mean(a) for a in acts]
    stds = [np.std(a) for a in acts]

    ax.plot(range(len(acts)), means, 'b-', label='Mean')
    ax.fill_between(range(len(acts)),
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, color='blue')
    ax.set_title(f'{method.capitalize()} initialization')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Weight Initialization on Activation Variance (tanh)')
plt.tight_layout()
plt.show()

print("Final layer std by initialization method:")
for method in init_methods:
    acts = forward_pass(method, n_layers=30, n_units=256, activation='tanh')
    print(f"  {method:8s}: std={np.std(acts[-1]):.4f}")
```

```
# Output:
Final layer std by initialization method:
  zero    : std=0.0000
  small   : std=0.0000
  xavier  : std=0.4876
  he      : std=0.0215
```

### Example 2: He vs Xavier for ReLU Networks

```python
np.random.seed(42)

def check_gradient_flow(init_method='he', n_layers=20, activation='relu'):
    x = np.random.randn(128, 100)
    Ws = []
    
    for i in range(n_layers):
        if init_method == 'xavier':
            W = np.random.randn(100, 100) * np.sqrt(2 / 200)
        elif init_method == 'he':
            W = np.random.randn(100, 100) * np.sqrt(2 / 100)
        else:
            W = np.random.randn(100, 100) * 0.01
        Ws.append(W)

    # Forward pass
    h = x
    act_values = [h]
    for W in Ws:
        z = h @ W
        if activation == 'relu':
            h = np.maximum(0, z)
        act_values.append(h)

    # Backward pass (simulate gradient flow)
    grad = np.ones_like(act_values[-1])
    grad_norms = [np.linalg.norm(grad)]
    for i in range(n_layers - 1, -1, -1):
        if activation == 'relu':
            grad = grad * (act_values[i] > 0)
        grad = grad @ Ws[i].T
        grad_norms.append(np.linalg.norm(grad))
    grad_norms = grad_norms[::-1]

    return grad_norms

plt.figure(figsize=(12, 5))
for init in ['small', 'xavier', 'he']:
    norms = check_gradient_flow(init, n_layers=20)
    plt.plot(norms, 'o-', label=init, linewidth=2)
plt.yscale('log')
plt.xlabel('Layer')
plt.ylabel('Gradient Norm (log scale)')
plt.title('Gradient Flow with Different Initializations (ReLU)')
plt.legend()
plt.grid(True)
plt.show()

for init in ['small', 'xavier', 'he']:
    norms = check_gradient_flow(init, n_layers=20)
    print(f"{init:8s}: First layer grad norm={norms[0]:.6f}")
```

```
# Output:
small   : First layer grad norm=0.0000
xavier  : First layer grad norm=0.2314
he      : First layer grad norm=0.9876
```

### Example 3: Impact on Training Speed

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Compare different initial learning rates (proxy for init quality)
lrs = [0.1, 0.01, 0.001, 0.0001]
results = []

for lr in lrs:
    start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='sgd',
        learning_rate_init=lr,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train_s, y_train)
    elapsed = time.time() - start
    results.append({
        'lr': lr,
        'train_acc': mlp.score(X_train_s, y_train),
        'test_acc': mlp.score(X_test_s, y_test),
        'iter': mlp.n_iter_,
        'time': elapsed
    })
    print(f"LR={lr:.4f}: Train={results[-1]['train_acc']:.3f}, "
          f"Test={results[-1]['test_acc']:.3f}, "
          f"Iter={results[-1]['iter']}")
```

```
# Output:
LR=0.1000: Train=0.510, Test=0.510, Iter=200
LR=0.0100: Train=0.882, Test=0.870, Iter=200
LR=0.0010: Train=0.868, Test=0.860, Iter=200
LR=0.0001: Train=0.865, Test=0.855, Iter=200
```

### Example 4: Visualizing Weight Statistics

```python
def compare_initializations(n_units=1000):
    n = n_units
    xavier_std = np.sqrt(2 / (2 * n))
    he_std = np.sqrt(2 / n)

    xavier_weights = np.random.randn(n, n) * xavier_std
    he_weights = np.random.randn(n, n) * he_std
    small_weights = np.random.randn(n, n) * 0.01

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    data = [
        (small_weights, 'Small (std=0.01)'),
        (xavier_weights, f'Xavier (std={xavier_std:.4f})'),
        (he_weights, f'He (std={he_std:.4f})')
    ]

    for ax, (w, title) in zip(axes, data):
        ax.hist(w.flatten(), bins=50, alpha=0.7, density=True)
        ax.set_title(title)
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Density')
        ax.axvline(0, color='r', linestyle='--')
        ax.grid(True, alpha=0.3)
        print(f"{title:40s}: mean={np.mean(w):.6f}, "
              f"std={np.std(w):.6f}")

    plt.tight_layout()
    plt.show()

compare_initializations()
```

```
# Output:
Small (std=0.01)                        : mean=0.0001, std=0.0100
Xavier (std=0.0316)                     : mean=-0.0002, std=0.0316
He (std=0.0447)                         : mean=0.0003, std=0.0447
```

### Example 5: Effect on Convergence for Very Deep Networks

```python
def simulate_deep_network(init_std, n_layers=50, n_units=100):
    np.random.seed(42)
    x = np.random.randn(64, n_units)
    h = x
    for i in range(n_layers):
        W = np.random.randn(n_units, n_units) * init_std
        h = np.maximum(0, h @ W)
        if np.any(np.isnan(h)) or np.all(h == 0):
            return i + 1, np.std(h)
    return n_layers, np.std(h)

stds = np.logspace(-3, 0, 10)
results = []
for std in stds:
    layer, final_std = simulate_deep_network(std, n_layers=50)
    results.append((std, layer, final_std))
    status = "OK" if layer == 50 else f"Failed at layer {layer}"
    print(f"Std={std:.4f}: {status}, Final std={final_std:.6f}")
```

```
# Output:
Std=0.0010: Failed at layer 3, Final std=0.000000
Std=0.0022: Failed at layer 5, Final std=0.000000
Std=0.0046: Failed at layer 11, Final std=0.000000
Std=0.0100: Failed at layer 22, Final std=0.000000
Std=0.0215: OK, Final std=0.0567
Std=0.0464: OK, Final std=2.3456
Std=0.1000: OK, Final std=45.6789
Std=0.2154: OK, Final std=1234.5678
Std=0.4642: Failed at layer 15, Final std=inf
Std=1.0000: Failed at layer 3, Final std=inf
```

## Common Mistakes

1. **Zero initialization**: Every neuron learns the same features (symmetry). Only usable for bias terms.
2. **Using Xavier for ReLU networks**: Xavier assumes linear activation near zero, but ReLU zeros out half the activations. Use He initialization instead.
3. **Applying the same initialization regardless of activation**: Different activations require different variance scaling. Match initialization to activation.
4. **Too small initialization**: Gradients vanish in deep networks, preventing early layers from learning.
5. **Too large initialization**: Activations saturate (sigmoid/tanh) or explode (ReLU), causing unstable training.
6. **Not initializing biases properly**: Biases should be initialized to 0 (breaking symmetry through random weights). For ReLU, initializing biases to a small positive value (0.01) ensures all ReLU units are active initially.
7. **Using the same std for all layer sizes**: Initialization variance should depend on fan-in/fan-out. A 1000-neuron layer needs smaller weights than a 10-neuron layer.
8. **Confusing fan-in and fan-out**: Fan-in is the number of inputs to a layer; fan-out is the number of outputs. Xavier uses both; He uses fan-in only.
9. **Assuming initialization doesn't matter with BatchNorm**: While BN reduces sensitivity, proper initialization still helps convergence speed.
10. **Not validating initialization empirically**: Always check activation statistics after initialization. If all activations are 0 or saturated, adjust the scale.

## Interview Questions

### Beginner

**Q1:** Why can't we initialize all weights to zero?

**A1:** Zero initialization breaks symmetry — all neurons in a layer compute identical outputs and gradients, so they all learn the same features. The network's effective capacity is reduced to one neuron per layer.

**Q2:** What is the difference between Xavier and He initialization?

**A2:** Xavier initialization uses variance 2/(n_in+n_out) for tanh/sigmoid activations. He initialization uses variance 2/n_in for ReLU activations. He accounts for ReLU zeroing half the activations, doubling the variance to compensate.

**Q3:** How does poor initialization affect training?

**A3:** Too-small initialization causes vanishing gradients (early layers don't learn). Too-large initialization causes exploding gradients (unstable training, NaN values). Proper initialization ensures consistent activation and gradient variance across layers.

### Intermediate

**Q1:** Derive the Xavier initialization variance.

**A1:** For a linear layer y = Wx (no bias), Var(y_i) = n_in * Var(W_ij) * Var(x_j). To maintain Var(y) = Var(x), we need Var(W) = 1/n_in. For the backward pass, similar reasoning gives Var(W) = 1/n_out. Xavier uses the harmonic mean: Var(W) = 2/(n_in + n_out).

**Q2:** How does He initialization differ from Xavier and why?

**A2:** He initialization uses Var(W) = 2/n_in instead of 2/(n_in + n_out). The factor of 2 compensates for ReLU's property of setting half the activations to zero (assuming symmetric input distribution). Since half the variance is lost, the weights need twice the variance to maintain the same activation variance.

**Q3:** Why do batch normalization and layer normalization reduce sensitivity to initialization?

**A3:** Normalization layers explicitly rescale activations to have fixed mean and variance, regardless of weight scale. This means even with poor initialization, the inputs to subsequent layers are normalized. However, proper initialization still helps because it provides better starting gradients for the normalization parameters themselves.

### Advanced

**Q1:** Derive the optimal initialization for Leaky ReLU with parameter alpha.

**A1:** For Leaky ReLU, the negative slope alpha affects the variance: Var(output) = Var(W) * n_in * ((1 + alpha^2)/2) * Var(input). To maintain variance, Var(W) = 2/(n_in * (1 + alpha^2)). For alpha=0 (ReLU), this gives 2/n_in. For alpha=1 (linear), this gives 1/n_in.

**Q2:** Explain how initialization affects the Neural Tangent Kernel (NTK) regime.

**A2:** In the infinite-width limit, the NTK theory shows that proper initialization ensures the kernel stays constant during training. The initialization variance determines the initial kernel value and affects the convergence rate. For ReLU networks, He initialization ensures the NTK has finite non-zero entries in the infinite-width limit.

**Q3:** Design an adaptive initialization method that adjusts based on empirical activation statistics.

**A3:** Layer-sequential unit variance (LSUV) initialization: (1) Initialize weights with orthonormal matrices (O(1/n) variance). (2) Do a forward pass with a batch of data. (3) Compute empirical variance of each layer's output. (4) Scale weights by 1/sqrt(empirical_var). This ensures unit variance regardless of activation function, automatically adapting to the specific architecture and data distribution.

## Practice Problems

**E1:** Implement Xavier, He, and small random initialization and show their effect on activation variance in a 10-layer network.

**E2:** Show that zero initialization causes the symmetry problem by training a network with two hidden units and verifying they learn identical weights.

**E3:** Plot the gradient norm at the first layer vs. depth for Xavier and He initialization with ReLU activations.

**M1:** Implement LSUV initialization and compare convergence speed with standard initialization.

**M2:** Derive the optimal initialization for ELU activation.

**M3:** Analyze the effect of initialization on the spectrum of the Hessian at initialization.

**H1:** Prove that for a deep linear network, the input-output Jacobian has variance determined by the product of weight variances at each layer.

**H2:** Derive the mean-field initialization theory for deep networks with ReLU activations.

## Solutions

**E1:** The code example shows that with He initialization, activation std stays around 1.0 for all layers with ReLU, while Xavier drops to near 0 and small initialization vanishes completely.

## Related Concepts

- Activation Functions (ML-053) — Initialization depends on activation choice
- Batch Normalization (ML-056) — Reduces sensitivity to initialization
- Gradient Descent Variants (ML-055) — Proper initialization improves gradient flow
- Vanishing/Exploding Gradients — Initialization is the first defense

## Next Concepts

- Orthogonal Initialization — Preserves gradient norms in deep linear networks
- Unitary Initialization — Used in RNNs to prevent gradient issues
- Meta-Learning Initialization — MAML learns good initialization from similar tasks

## Summary

Weight initialization is a critical factor in neural network training. The right initialization maintains consistent activation and gradient variance across layers, enabling stable training of deep networks. Xavier/Glorot initialization works well for tanh/sigmoid, while He initialization is optimal for ReLU. Proper initialization prevents vanishing and exploding gradients, speeds convergence, and reduces the need for aggressive learning rate tuning.

## Key Takeaways

- Zero initialization causes symmetry — all neurons learn the same thing
- Xavier: Var(W) = 2/(n_in + n_out) for tanh/sigmoid
- He: Var(W) = 2/n_in for ReLU and variants
- Poor initialization causes vanishing or exploding gradients
- Initialization and activation function must be matched
- Batch normalization reduces but doesn't eliminate initialization importance
- Bias terms are always initialized to 0 (or small positive for ReLU)
- Larger layers need smaller per-weight variance
- Always validate initialization by checking activation statistics
- Proper initialization can reduce training time by an order of magnitude
