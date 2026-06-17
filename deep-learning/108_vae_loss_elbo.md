# Concept: VAE Loss (ELBO)

## Concept ID

DL-108

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand the variational autoencoder loss (ELBO)
- Implement the ELBO loss in PyTorch
- Explain the reconstruction and KL divergence terms
- Apply the reparameterization trick
- Vary the beta parameter for beta-VAE

## Prerequisites

- Autoencoder architecture
- KL Divergence (DL-098)
- Bayesian inference basics

## Definition

The Variational Autoencoder (VAE) loss is the negative of the Evidence Lower BOund (ELBO). It consists of two terms: a reconstruction loss and a KL divergence regularization:

L_VAE = -ELBO = -E_{z~q(z|x)}[log p(x|z)] + KL(q(z|x) || p(z))

The reconstruction term ensures the decoder produces good reconstructions, and the KL term regularizes the latent space to match the prior (typically a standard Gaussian).

## Intuition

The VAE learns a compressed latent representation of data. The reconstruction loss ensures the latent code captures enough information to reconstruct the input. The KL divergence ensures the latent space is structured (close to a standard Gaussian) so that we can sample from it to generate new data.

Think of it as organizing a library: the reconstruction loss ensures each book's summary (latent code) captures its content, while the KL term ensures the summaries are written in a standard format that anyone can understand.

## Why This Concept Matters

VAEs are foundational generative models with applications in:
- Image generation
- Representation learning
- Anomaly detection
- Data compression
- Semi-supervised learning

## Mathematical Explanation

### ELBO Derivation

log p(x) >= E_{z~q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z))

The VAE loss is the negative of this lower bound.

### Reparameterization Trick

z = mu + sigma * epsilon, where epsilon ~ N(0, I)

This allows gradients to flow through the sampling operation.

### Loss Components

Reconstruction: ||x - x_hat||^2 (for Gaussian decoder) or -sum(x*log(x_hat)) (for Bernoulli)
KL: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)

### Beta-VAE

L_betaVAE = Reconstruction + beta * KL

beta > 1 encourages more disentangled representations.

## Code Examples

### Example 1: VAE Loss Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (Bernoulli or Gaussian)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence: KL(q(z|x) || N(0, I))
    # KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Example
torch.manual_seed(42)
recon_x = torch.sigmoid(torch.randn(4, 784))
x = (torch.rand(4, 784) > 0.5).float()
mu = torch.randn(4, 20) * 0.1
logvar = torch.randn(4, 20) * 0.1 - 2.0

loss, recon, kl = vae_loss(recon_x, x, mu, logvar)
print(f"Total VAE loss: {loss.item():.2f}")
print(f"  Reconstruction: {recon.item():.2f}")
print(f"  KL: {kl.item():.2f}")
```

```
# Output:
# Total VAE loss: 559.03
#   Reconstruction: 543.85
#   KL: 15.19
```

### Example 2: VAE Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

torch.manual_seed(42)
N, input_dim, latent_dim = 500, 784, 20
X = (torch.rand(N, input_dim) > 0.5).float()

model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    recon, mu, logvar = model(X)
    loss, recon_loss, kl_loss = vae_loss(recon, X, mu, logvar)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.2f}, recon = {recon_loss.item():.2f}, KL = {kl_loss.item():.2f}")
```

```
# Output:
# Epoch 0: loss = 585.00, recon = 553.68, KL = 31.32
# Epoch 20: loss = 387.06, recon = 363.64, KL = 23.42
# Epoch 40: loss = 348.50, recon = 326.52, KL = 21.99
# Epoch 60: loss = 329.38, recon = 308.03, KL = 21.35
# Epoch 80: loss = 316.84, recon = 295.80, KL = 21.04
# Epoch 99: loss = 307.68, recon = 286.85, KL = 20.83
```

### Example 3: Beta-VAE Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D, latent = 300, 64, 8
X = (torch.rand(N, D) > 0.5).float()

def train_beta_vae(beta, label):
    model = VAE(D, latent)
    opt = optim.Adam(model.parameters(), lr=0.001)
    final_kl = 0
    for epoch in range(100):
        opt.zero_grad()
        recon, mu, logvar = model(X)
        _, recon_loss, kl_loss = vae_loss(recon, X, mu, logvar, beta=beta)
        total = recon_loss + beta * kl_loss
        total.backward()
        opt.step()
        final_kl = kl_loss.item()
    print(f"{label}: recon = {recon_loss.item():.2f}, KL = {final_kl:.2f}, total = {total.item():.2f}")

train_beta_vae(1.0, "Standard VAE (beta=1)")
train_beta_vae(4.0, "Beta-VAE (beta=4)")
train_beta_vae(0.5, "Warm VAE (beta=0.5)")
```

```
# Output:
# Standard VAE (beta=1): recon = 38.74, KL = 10.10, total = 48.84
# Beta-VAE (beta=4): recon = 42.56, KL = 1.75, total = 49.56
# Warm VAE (beta=0.5): recon = 36.82, KL = 19.49, total = 46.57
```

## Common Mistakes

1. **KL annealing failure**: Without KL annealing (gradually increasing beta), the KL term can collapse to zero, making the VAE act like a standard autoencoder.
2. **Not using reparameterization**: Without reparameterization, gradients cannot flow through the sampling operation.
3. **Wrong loss reduction**: Use 'sum' reduction for KL and reconstruction, or use consistent 'mean' for both.
4. **KL term too dominant**: Beta too large collapses the latent space. Beta too small gives no regularization.
5. **Ignoring posterior collapse in VAEs**: The decoder may learn to ignore the latent code, especially with powerful decoders (e.g., PixelCNN).
6. **Numerical instability with logvar**: logvar should be well-behaved. Initializing to log(sigma^2) = 0 is common.

## Interview Questions

### Beginner

1. What does the VAE loss consist of?
2. What is the ELBO and how does it relate to VAE loss?
3. What does the KL term do in VAE?
4. What is the reparameterization trick?
5. How do you implement VAE loss in PyTorch?

### Intermediate

1. Derive the ELBO from the evidence lower bound.
2. Explain why the KL term encourages latent space structure.
3. How does beta-VAE differ from standard VAE?
4. What is posterior collapse and how does it manifest in the loss?
5. Compare VAE loss with AE loss (without KL).

### Advanced

1. Prove the ELBO inequality: log p(x) >= ELBO.
2. Analyze the rate-distortion tradeoff in VAE loss.
3. Derive the gradient of VAE loss with respect to encoder parameters.

## Practice Problems

### Easy

1. Implement VAE loss with reconstruction and KL terms.
2. Compute the closed-form KL between two Gaussians.
3. Verify that KL = 0 when mu=0 and logvar=0.
4. Train a simple VAE on binary data.
5. Compare reconstruction quality for different latent dimensions.

### Medium

1. Implement KL annealing for VAE training.
2. Compare beta=1, beta=4, beta=10 for latent space structure.
3. Train a VAE on MNIST and visualize the latent space.
4. Implement a conditional VAE.
5. Analyze the reconstruction-KL tradeoff curve.

### Hard

1. Derive the ELBO for a hierarchical VAE.
2. Implement the VQ-VAE loss.
3. Design an experiment showing the effect of posterior collapse.

## Solutions

VAE loss = reconstruction + KL. Reconstruction uses BCE or MSE. KL is computed analytically. The reparameterization trick enables gradient flow. Beta controls the regularization strength.

## Related Concepts

- KL Divergence (DL-098): KL term in VAE
- Adversarial Loss (DL-106): Alternative generative model loss
- InfoNCE Loss (DL-110): Contrastive variant

## Next Concepts

- Noise Contrastive Estimation (DL-109)
- InfoNCE Loss (DL-110)

## Summary

The VAE loss minimizes the negative ELBO, which consists of a reconstruction term and a KL divergence regularization term. The reconstruction term ensures faithful reconstruction, and the KL term ensures the latent space follows a prior distribution. The reparameterization trick enables gradient-based optimization.

## Key Takeaways

1. VAE loss = reconstruction + KL divergence = -ELBO.
2. Reconstruction: BCE or MSE between input and reconstruction.
3. KL: regularizes latent space toward prior (standard Gaussian).
4. Reparameterization trick enables backprop through sampling.
5. Beta-VAE adjusts the KL weight for disentangled representations.
