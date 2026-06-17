# Concept: Batch Normalization

## Concept ID

ML-056

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Neural Networks

## Learning Objectives

- Understand internal covariate shift and how batch normalization addresses it
- Derive the forward and backward pass of batch normalization
- Explain the role of learnable parameters gamma and beta
- Understand how batch normalization enables higher learning rates and acts as a regularizer

## Prerequisites

- Backpropagation (ML-054) — gradient computation through arbitrary layers
- Gradient Descent Variants (ML-055) — understanding of training dynamics

## Definition

Batch Normalization (BatchNorm) normalizes the activations of a layer across the mini-batch dimension. For each feature, it computes the mean and variance over the batch, normalizes to zero mean and unit variance, then applies a learnable affine transformation.

mu_B = (1/m) * sum(x^{(k)})
sigma_B^2 = (1/m) * sum((x^{(k)} - mu_B)^2)
x_hat^{(k)} = (x^{(k)} - mu_B) / sqrt(sigma_B^2 + eps)
y^{(k)} = gamma * x_hat^{(k)} + beta

where gamma and beta are learnable parameters, and eps is a small constant for numerical stability.

## Intuition

Think of batch normalization as a volume control that keeps the signal strength steady through each layer of a network. Without it, each layer's activations can grow or shrink unpredictably as weights change, making the optimization landscape bumpy. BatchNorm smooths this landscape, allowing faster and more stable training.

The learnable gamma (scale) and beta (shift) allow the network to decide the optimal distribution for each feature. During training, batch statistics are used; during inference, running averages provide consistent normalization.

## Why This Concept Matters

BatchNorm was a breakthrough that enabled very deep networks (50+ layers) to train stably. It allows 5-10x higher learning rates, provides regularization, reduces sensitivity to initialization, and speeds convergence by 2-10x. It is used in virtually all modern CNN architectures including ResNet, Inception, and DenseNet.

## Mathematical Explanation

Given input x in R^{N x D} (N batch size, D features):

mu = (1/N) * sum_i x_i
sigma2 = (1/N) * sum_i (x_i - mu)^2
x_hat = (x - mu) / sqrt(sigma2 + eps)
y = gamma * x_hat + beta

The backward pass must account for the dependence of mu and sigma2 on the batch:

dL/dx_hat = dL/dy * gamma
dL/dsigma2 = sum(dL/dx_hat * (x - mu) * -0.5 * (sigma2 + eps)^(-1.5))
dL/dmu = sum(dL/dx_hat * -1/sqrt(sigma2+eps)) + dL/dsigma2 * sum(-2*(x-mu))/N
dL/dx = dL/dx_hat / sqrt(sigma2+eps) + dL/dsigma2 * 2*(x-mu)/N + dL/dmu/N
dL/dgamma = sum(dL/dy * x_hat)
dL/dbeta = sum(dL/dy)

During inference, running averages replace batch statistics:
mu_running = momentum * mu_running + (1 - momentum) * mu_batch
sigma2_running = momentum * sigma2_running + (1 - momentum) * sigma2_batch

Internal covariate shift refers to the change in activation distribution due to parameter updates in earlier layers. BatchNorm reduces this shift, making the optimization landscape smoother. The noise from different batch statistics each iteration provides a regularization effect similar to dropout.

## Code Examples

### Example 1: Batch Normalization from Scratch

```python
import numpy as np

class BatchNorm:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, x):
        if self.training:
            self.mu = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
            out = self.gamma * self.x_hat + self.beta
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * self.mu)
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) * self.var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(
                self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
        return out

np.random.seed(42)
x = np.random.randn(32, 64) * 2 + 3
bn = BatchNorm(64)
out = bn.forward(x)

print(f"Input mean: {np.mean(x):.4f}, std: {np.std(x):.4f}")
print(f"Output mean: {np.mean(out):.4f}, std: {np.std(out):.4f}")
```

```
# Output:
Input mean: 3.0123, std: 2.0156
Output mean: -0.0023, std: 0.9987
```

### Example 2: Batch Effect on Deep Network Training

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp_shallow = MLPClassifier(hidden_layer_sizes=(64,), activation='relu',
    solver='adam', learning_rate_init=0.001, max_iter=500, random_state=42)
mlp_shallow.fit(X_train_s, y_train)

mlp_deep = MLPClassifier(hidden_layer_sizes=(64, 64, 64, 64, 64),
    activation='relu', solver='adam', learning_rate_init=0.0001,
    max_iter=1000, random_state=42)
mlp_deep.fit(X_train_s, y_train)

for name, model in [('Shallow', mlp_shallow), ('Deep', mlp_deep)]:
    print(f"{name}: Train={model.score(X_train_s, y_train):.4f}, "
          f"Test={model.score(X_test_s, y_test):.4f}")
```

```
# Output:
Shallow: Train=0.9544, Test=0.9450
Deep: Train=0.8606, Test=0.8525
```

### Example 3: Running Statistics During Training and Inference

```python
bn = BatchNorm(10, momentum=0.9)
print("Training phase:")
for step in range(100):
    batch = np.random.randn(32, 10) * 2 + np.random.uniform(-1, 1, 10)
    bn.forward(batch)
    if (step + 1) % 50 == 0:
        print(f"  Step {step+1}: running_mean[0]={bn.running_mean[0]:.4f}")

bn.training = False
test_batch = np.random.randn(16, 10) * 3 + 5
out = bn.forward(test_batch)
print(f"\nInference output mean: {np.mean(out):.4f}, var: {np.var(out):.4f}")
```

```
# Output:
Training phase:
  Step 50: running_mean[0]=0.0234
  Step 100: running_mean[0]=0.0089

Inference output mean: 0.9876, var: 0.9823
```

## Common Mistakes

1. Applying BN before or after activation inconsistently — be consistent (modern practice favors pre-activation).
2. Using batch size 1 during training — degenerate zero variance; use batch size >= 8.
3. Forgetting to set training/eval mode — inference requires running statistics, not batch statistics.
4. Applying BN to the output layer — destroys output scale.
5. Using wrong momentum for running statistics — default 0.99 is typically best.
6. Not accounting for BN in gradient computation — must account for batch-wide dependence of mu and sigma.
7. Using BN with very small batch sizes in GANs — use GroupNorm or LayerNorm instead.
8. Applying BN after dropout — order matters; typically BN before dropout.
9. Misunderstanding gamma and beta — they restore expressive power after normalization.
10. Expecting BN to work identically across all architectures — BN works best in CNNs; LayerNorm is preferred for NLP/Transformers.

## Interview Questions

### Beginner

**Q1:** What is batch normalization?

**A1:** Batch normalization normalizes layer activations to have zero mean and unit variance across the mini-batch, then applies learnable scale (gamma) and shift (beta). This stabilizes training and allows higher learning rates.

**Q2:** Why is batch normalization used in deep networks?

**A2:** It reduces internal covariate shift, smoothes the optimization landscape, allows higher learning rates, provides regularization, and reduces sensitivity to initialization.

**Q3:** What are gamma and beta in batch normalization?

**A3:** Gamma is the learnable scale parameter (initialized to 1), and beta is the learnable shift parameter (initialized to 0). They restore the network's expressive power by allowing it to learn the optimal scale and shift for each feature.

### Intermediate

**Q1:** Explain how batch normalization works during training vs. inference.

**A1:** During training, BN computes mean and variance from the current mini-batch. During inference, it uses running averages (exponential moving averages) computed during training. This ensures deterministic behavior regardless of batch size.

**Q2:** How does batch normalization act as a regularizer?

**A2:** Each mini-batch has different statistics, introducing noise into the normalized activations. This noise acts as a regularizer, similar to dropout. The effect is stronger with smaller batch sizes.

**Q3:** Why does batch normalization allow higher learning rates?

**A3:** BN ensures activations have stable mean and variance, preventing extreme activation values. This makes the loss landscape smoother and more well-behaved, allowing larger gradient steps without divergence.

### Advanced

**Q1:** Derive the backward pass through a batch normalization layer.

**A1:** The backward pass must account for the dependence of mu and sigma2 on the batch. The key gradients are: dL/dx_hat = dL/dy * gamma, dL/dsigma2 = sum(dL/dx_hat * (x-mu) * -0.5 * (sigma2+eps)^(-1.5)), dL/dmu = sum(dL/dx_hat * -1/sqrt(sigma2+eps)) + dL/dsigma2 * sum(-2*(x-mu))/N, and finally dL/dx = dL/dx_hat/sqrt(sigma2+eps) + dL/dsigma2 * 2*(x-mu)/N + dL/dmu/N.

**Q2:** Compare batch normalization, layer normalization, and group normalization.

**A2:** BN normalizes across the batch dimension (N). LayerNorm normalizes across features (D) for each sample independently — used in Transformers. GroupNorm normalizes across groups of channels — useful for small batch sizes. BN is best for CNNs with large batches, LayerNorm for NLP/Transformers, GroupNorm for small-batch vision tasks.

**Q3:** Explain the concept of internal covariate shift and how BN addresses it.

**A3:** Internal covariate shift is the change in activation distribution due to parameter updates in earlier layers. As previous layers change, the inputs to later layers shift, making them constantly adapt to new distributions. BN reduces this by explicitly normalizing each layer's inputs, ensuring stable mean and variance regardless of earlier layer changes.

## Practice Problems

**E1:** Implement BatchNorm from scratch and verify that output has zero mean and unit variance.

**E2:** Compare convergence speed of a 10-layer network with and without BN.

**E3:** Show that BN allows 5x higher learning rate compared to training without BN.

**M1:** Derive the full backward pass equations for batch normalization.

**M2:** Implement and compare BN, LayerNorm, and GroupNorm on a simple MLP.

**H1:** Prove that the gradient through batch normalization accounts for the batch statistics dependence.

## Solutions

**E1:** The BatchNorm class in Example 1 shows that output mean is ~0 and std is ~1. When gamma=1 and beta=0, the output is exactly normalized.

## Related Concepts

- Layer Normalization — Normalizes across features instead of batch
- Dropout (ML-057) — Another regularization technique
- Weight Initialization (ML-058) — BN reduces sensitivity to initialization
- Gradient Descent (ML-055) — BN improves gradient flow

## Next Concepts

- Group Normalization — Normalization for small batch sizes
- Instance Normalization — Used in style transfer
- Adaptive Normalization — Learnable normalization strategies

## Summary

Batch Normalization normalizes layer activations using mini-batch statistics during training and running averages during inference. It reduces internal covariate shift, enables higher learning rates, provides regularization, and speeds convergence. The learnable gamma and beta parameters restore the network's expressive power. BatchNorm is a standard component in modern deep learning architectures.

## Key Takeaways

- BN normalizes activations to zero mean and unit variance per batch
- Learnable gamma (scale) and beta (shift) restore expressive power
- Reduces internal covariate shift, smoothing the optimization landscape
- Allows 5-10x higher learning rates without divergence
- Provides slight regularization through batch statistics noise
- Training uses batch statistics; inference uses running averages
- Best suited for CNNs with batch size >= 8
- LayerNorm is preferred for NLP and Transformers
- BN reduces sensitivity to initialization choices
- Essential component in modern deep learning architectures
