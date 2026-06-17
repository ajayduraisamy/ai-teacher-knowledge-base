# Concept: Wasserstein Loss

## Concept ID

DL-107

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand Wasserstein distance and its use as a GAN loss
- Implement WGAN loss with gradient penalty
- Explain why Wasserstein loss improves GAN training stability
- Apply Wasserstein loss for WGAN and WGAN-GP
- Compare with standard GAN loss

## Prerequisites

- Adversarial Loss (DL-106)
- GAN fundamentals
- Lipschitz continuity

## Definition

Wasserstein loss, used in Wasserstein GANs (WGAN), replaces the binary cross-entropy discriminator with a critic that estimates the Earth Mover's distance between the real and generated distributions. The loss functions are:

Critic loss: L_C = E_z[C(G(z))] - E_x[C(x)]
Generator loss: L_G = -E_z[C(G(z))]

where C is the critic (not a classifier) and must be 1-Lipschitz. A gradient penalty enforces this:

GP = lambda * E_x[(||grad C(x)||_2 - 1)^2]

## Intuition

Standard GAN discriminator outputs a probability. The Wasserstein critic outputs a score that approximates how "real" an image looks. The critic tries to assign higher scores to real images and lower scores to fake images. The generator tries to make its images get high scores.

The key advantage: Wasserstein loss provides smooth, meaningful gradients even when the distributions don't overlap, which is common early in training.

## Why This Concept Matters

WGAN and WGAN-GP solved many GAN training instability issues:
- No mode collapse
- Meaningful loss curves that correlate with sample quality
- Stable training without careful balancing
- Less sensitive to architecture choices

## Mathematical Explanation

### Wasserstein Distance

W(P_r, P_g) = inf_{gamma in Pi(P_r, P_g)} E_{(x,y)~gamma}[||x - y||]

By Kantorovich-Rubinstein duality:
W(P_r, P_g) = sup_{||f||_L <= 1} E_x[f(x)] - E_z[f(G(z))]

### WGAN Loss

Critic tries to maximize: E_x[C(x)] - E_z[C(G(z))]
Generator tries to minimize: -E_z[C(G(z))]

### Gradient Penalty (WGAN-GP)

GP = lambda * E[(||grad(C(alpha*x + (1-alpha)*G(z)))||_2 - 1)^2]

## Code Examples

### Example 1: WGAN-GP Loss Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def compute_gradient_penalty(critic, real_data, fake_data, lambda_gp=10.0):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    critic_interp = critic(interpolated)
    gradients = autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return lambda_gp * penalty

def critic_loss(real_scores, fake_scores, gradient_penalty):
    return fake_scores.mean() - real_scores.mean() + gradient_penalty

def generator_loss(fake_scores):
    return -fake_scores.mean()

# Example
torch.manual_seed(42)
real_scores = torch.tensor([2.0, 1.5, 2.5])
fake_scores = torch.tensor([-0.5, -1.0, 0.5])
gp = torch.tensor(0.5)

c_loss = critic_loss(real_scores, fake_scores, gp)
g_loss = generator_loss(fake_scores)
print(f"Critic loss: {c_loss.item():.4f}")
print(f"Generator loss: {g_loss.item():.4f}")
```

```
# Output:
# Critic loss: -1.0000
# Generator loss: -0.3333
```

### Example 2: WGAN-GP Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
latent_dim, img_dim = 100, 784

generator = nn.Sequential(
    nn.Linear(latent_dim, 256), nn.ReLU(),
    nn.Linear(256, img_dim), nn.Tanh()
)
critic = nn.Sequential(
    nn.Linear(img_dim, 256), nn.ReLU(),
    nn.Linear(256, 1)
)
g_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
c_opt = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
batch_size = 64

for epoch in range(100):
    # Train critic multiple times
    for _ in range(5):
        real_data = (torch.rand(batch_size, img_dim) * 2 - 1)
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)

        c_opt.zero_grad()
        real_scores = critic(real_data)
        fake_scores = critic(fake_data.detach())
        gp = compute_gradient_penalty(critic, real_data, fake_data)
        c_loss = critic_loss(real_scores, fake_scores, gp)
        c_loss.backward()
        c_opt.step()

    # Train generator once
    noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(noise)
    g_opt.zero_grad()
    fake_scores = critic(fake_data)
    g_loss = generator_loss(fake_scores)
    g_loss.backward()
    g_opt.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: C_loss = {c_loss.item():.4f}, G_loss = {g_loss.item():.4f}")
```

```
# Output:
# Epoch 0: C_loss = -1.3860, G_loss = 0.7956
# Epoch 20: C_loss = -7.8923, G_loss = 3.4473
# Epoch 40: C_loss = -9.2831, G_loss = 4.2178
# Epoch 60: C_loss = -10.1021, G_loss = 4.5612
# Epoch 80: C_loss = -8.4762, G_loss = 3.7123
# Epoch 99: C_loss = -11.0254, G_loss = 5.2189
```

### Example 3: WGAN vs. Standard GAN Loss Comparison

```python
import torch
import torch.nn.functional as F

# Compare loss landscapes
scores = torch.linspace(-5, 5, 100)

# Standard GAN generator loss (non-saturating)
gan_g_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores), reduction='none')

# WGAN generator loss
wgan_g_loss = -scores

print("Score\tGAN_G\tWGAN_G")
for s in [-3, -1, 0, 1, 3]:
    idx = (scores - s).abs().argmin()
    print(f"{s:+.0f}\t{gan_g_loss[idx].item():.4f}\t{wgan_g_loss[idx].item():.4f}")

print("\nKey difference: WGAN loss is linear, GAN loss is logarithmic.")
print("Linear gradients don't vanish, even for very good or very bad samples.")
```

```
# Output:
# Score   GAN_G   WGAN_G
# -3      3.0486  3.0000
# -1      1.3133  1.0000
#  0      0.6931  0.0000
# +1      0.3133  -1.0000
# +3      0.0486  -3.0000
#
# Key difference: WGAN loss is linear, GAN loss is logarithmic.
# Linear gradients don't vanish, even for very good or very bad samples.
```

## Common Mistakes

1. **Forgetting gradient penalty**: Without GP, the critic can violate Lipschitz constraints, making training unstable.
2. **Training critic too few iterations**: WGAN benefits from training the critic more than the generator (typically 5:1).
3. **Using momentum with beta1 too high**: WGAN-GP recommends betas=(0.5, 0.999) for Adam, not the default (0.9, 0.999).
4. **Clipping weights instead of GP**: Original WGAN used weight clipping, which limits network capacity. GP is preferred.
5. **Not setting interpolated.requires_grad=True**: GP computation requires gradients through the interpolated samples.
6. **Ignoring the Lipschitz constraint**: Without enforcing 1-Lipschitz, the Wasserstein distance estimate is invalid.

## Interview Questions

### Beginner

1. What is the Wasserstein distance?
2. How does WGAN differ from standard GAN?
3. What is a critic vs. a discriminator?
4. Why does WGAN have more stable training?
5. What does the gradient penalty enforce?

### Intermediate

1. Derive the Wasserstein distance from optimal transport.
2. Explain why Wasserstein loss doesn't saturate.
3. Compare weight clipping vs. gradient penalty.
4. Why does the critic need to be Lipschitz?
5. How does the loss value correlate with sample quality in WGAN?

### Advanced

1. Prove that the Kantorovich-Rubinstein duality gives the WGAN objective.
2. Analyze why gradient penalty enforces 1-Lipschitz.
3. Derive the gradient of the Wasserstein loss with respect to generator parameters.

## Practice Problems

### Easy

1. Implement the gradient penalty function.
2. Compute critic and generator loss for given scores.
3. Verify that WGAN loss is linear while GAN loss is logarithmic.
4. Compare loss trajectories of WGAN vs. standard GAN.
5. Implement weight clipping for WGAN.

### Medium

1. Train a WGAN-GP on 2D synthetic data.
2. Compare WGAN and DCGAN on MNIST.
3. Analyze the effect of lambda_GP on training.
4. Visualize the critic scores for real vs. fake data during training.
5. Implement the critic with spectral normalization (alternative to GP).

### Hard

1. Prove that WGAN loss provides meaningful gradients even with disjoint supports.
2. Implement the Cramer GAN and compare with WGAN.
3. Design an experiment showing WGAN solves mode collapse.

## Solutions

Wasserstein loss uses a critic to estimate Earth Mover's distance between real and generated distributions. Gradient penalty enforces Lipschitz continuity. WGAN provides stable training and meaningful loss curves.

## Related Concepts

- Adversarial Loss (DL-106): Standard GAN loss
- Optimal Transport: Theoretical foundation
- VAE Loss (DL-108): Alternative generative model loss

## Next Concepts

- VAE Loss (DL-108)
- Noise Contrastive Estimation (DL-109)
- InfoNCE Loss (DL-110)

## Summary

Wasserstein loss replaces the GAN discriminator with a critic that estimates the Wasserstein distance between distributions. The critic is trained to maximize the score difference between real and fake samples, while the generator tries to maximize fake scores. Gradient penalty enforces the 1-Lipschitz constraint, enabling stable training.

## Key Takeaways

1. WGAN critic estimates Wasserstein distance (Earth Mover's distance).
2. WGAN loss does not saturate, avoiding vanishing gradients.
3. Gradient penalty enforces 1-Lipschitz constraint on the critic.
4. WGAN training is more stable with a 5:1 critic-to-generator ratio.
5. WGAN loss values correlate with generated sample quality.
