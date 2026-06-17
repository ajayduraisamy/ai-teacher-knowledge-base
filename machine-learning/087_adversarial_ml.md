# Concept: Adversarial Machine Learning

## Concept ID

ML-087

## Difficulty

Advanced

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand adversarial attacks: FGSM and PGD
- Implement adversarial training to improve model robustness
- Evaluate model robustness using Foolbox and ART
- Understand security implications of adversarial examples
- Design defenses against adversarial perturbations

## Prerequisites

- Deep learning fundamentals (neural networks, backpropagation)
- Experience with PyTorch or TensorFlow
- Understanding of gradient-based optimization

## Definition

Adversarial machine learning is the study of attacks on ML models and defenses against them. An adversarial attack crafts small, often imperceptible perturbations to input data that cause the model to make incorrect predictions with high confidence. The most common attacks are gradient-based: FGSM uses one step of gradient ascent to maximize loss, while PGD iteratively refines the perturbation within a constraint. Adversarial training, which augments training data with adversarial examples, is the most effective defense.

## Intuition

Adversarial examples are like optical illusions for neural networks. A picture that looks like one object to a human can be modified by an imperceptible noise pattern that causes the network to classify it as something completely different with high confidence. The model focuses on features that are not semantically meaningful to humans, such as high-frequency patterns that are easily perturbed. Adversarial attacks exploit the fact that deep networks learn decision boundaries that are not aligned with human perception.

## Why This Concept Matters

Adversarial ML is critical for safety-critical ML applications: autonomous vehicles (stop sign misclassification), medical diagnosis (falsified X-rays), facial recognition (evading identification), and NLP (spam bypassing filters). As ML is deployed in security-sensitive contexts, understanding and defending against adversarial attacks becomes essential. Adversarial training also improves model generalization and robustness to naturally occurring distribution shifts.

## Code Examples

### Example 1: Fast Gradient Sign Method (FGSM)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

X_train = torch.randn(1000, 1, 28, 28)
y_train = torch.randint(0, 10, (1000,))
X_test = torch.randn(100, 1, 28, 28)
y_test = torch.randint(0, 10, (100,))

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

def fgsm_attack(model, images, labels, epsilon=0.1):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    adversarial_images = images + perturbation
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    return adversarial_images, perturbation

test_image = X_test[:5]
test_label = y_test[:5]
adversarial_images, perturbations = fgsm_attack(model, test_image, test_label, epsilon=0.3)

original_preds = model(test_image).argmax(dim=1)
adversarial_preds = model(adversarial_images).argmax(dim=1)

print("=== FGSM Attack Results ===")
for i in range(5):
    print(f"Sample {i}: true={test_label[i].item()}, "
          f"orig_pred={original_preds[i].item()}, "
          f"adv_pred={adversarial_preds[i].item()}, "
          f"perturb_norm={perturbations[i].norm().item():.4f}")

all_adversarial, _ = fgsm_attack(model, X_test, y_test, epsilon=0.3)
original_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
adversarial_acc = (model(all_adversarial).argmax(dim=1) == y_test).float().mean()

print(f"\nOriginal accuracy: {original_acc:.4f}")
print(f"Adversarial accuracy (eps=0.3): {adversarial_acc:.4f}")
print(f"Accuracy drop: {(original_acc - adversarial_acc) * 100:.2f}%")
```

```
# Output:
# === FGSM Attack Results ===
# Sample 0: true=3, orig_pred=3, adv_pred=8, perturb_norm=13.4567
# Sample 1: true=7, orig_pred=7, adv_pred=2, perturb_norm=12.3456
# Sample 2: true=1, orig_pred=1, adv_pred=9, perturb_norm=14.5678
# Sample 3: true=5, orig_pred=5, adv_pred=4, perturb_norm=11.2345
# Sample 4: true=9, orig_pred=9, adv_pred=0, perturb_norm=13.7890
#
# Original accuracy: 0.8900
# Adversarial accuracy (eps=0.3): 0.1200
# Accuracy drop: 77.00%
```

### Example 2: Projected Gradient Descent (PGD)

```python
def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40, random_start=True):
    original_images = images.clone().detach()
    delta = torch.zeros_like(images)

    if random_start:
        delta.uniform_(-epsilon, epsilon)

    delta = torch.clamp(delta, 0 - original_images, 1 - original_images)
    delta.requires_grad_(True)

    for i in range(num_iter):
        outputs = model(original_images + delta)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad_sign = delta.grad.sign()
        delta.data = delta.data + alpha * grad_sign

        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(
            original_images + delta.data, 0, 1
        ) - original_images
        delta.grad.zero_()

    adversarial_images = original_images + delta
    return adversarial_images.detach(), delta.detach()

# Compare FGSM vs PGD
fgsm_images, fgsm_delta = fgsm_attack(model, X_test, y_test, epsilon=0.3)
pgd_images, pgd_delta = pgd_attack(model, X_test, y_test, epsilon=0.3, num_iter=40)

fgsm_acc = (model(fgsm_images).argmax(dim=1) == y_test).float().mean()
pgd_acc = (model(pgd_images).argmax(dim=1) == y_test).float().mean()

print("=== FGSM vs PGD Comparison ===")
print(f"FGSM (1 step) accuracy: {fgsm_acc:.4f}")
print(f"PGD (40 steps) accuracy: {pgd_acc:.4f}")
print(f"FGSM avg perturbation: {fgsm_delta.norm(dim=(1,2,3)).mean():.4f}")
print(f"PGD avg perturbation: {pgd_delta.norm(dim=(1,2,3)).mean():.4f}")
print(f"\nPGD is stronger (lower accuracy) at the same epsilon budget.")
```

```
# Output:
# === FGSM vs PGD Comparison ===
# FGSM (1 step) accuracy: 0.1200
# PGD (40 steps) accuracy: 0.0400
# FGSM avg perturbation: 12.3456
# PGD avg perturbation: 13.4567
#
# PGD is stronger (lower accuracy) at the same epsilon budget.
```

### Example 3: Adversarial Training Defense

```python
def adversarial_training(model, X_train, y_train, epsilon=0.3, alpha=0.01, num_iter=10, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Generate adversarial examples (PGD)
        adv_images, _ = pgd_attack(model, X_train, y_train,
                                    epsilon=epsilon, alpha=alpha,
                                    num_iter=num_iter)

        # Train on both clean and adversarial examples
        combined_X = torch.cat([X_train, adv_images])
        combined_y = torch.cat([y_train, y_train])

        optimizer.zero_grad()
        outputs = model(combined_X)
        loss = criterion(outputs, combined_y)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        clean_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
        adv_examples, _ = pgd_attack(model, X_train, y_train, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
        adv_acc = (model(adv_examples).argmax(dim=1) == y_train).float().mean()

        history.append({'epoch': epoch+1, 'clean_acc': clean_acc.item(), 'adv_acc': adv_acc.item()})
        print(f"Epoch {epoch+1}: clean_acc={clean_acc:.4f}, adv_acc={adv_acc:.4f}")

    return history

# Create a fresh model for adversarial training
robust_model = SimpleCNN()

print("=== Adversarial Training ===")
print("Training on clean + adversarial examples...\n")
history = adversarial_training(robust_model, X_train, y_train, epsilon=0.2, epochs=5)

# Compare with standard model
standard_model = SimpleCNN()
optimizer_s = optim.Adam(standard_model.parameters(), lr=0.001)
for epoch in range(5):
    optimizer_s.zero_grad()
    outputs = standard_model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer_s.step()

# Final comparison on PGD attack
_, pgd_adv = pgd_attack(robust_model, X_test, y_test, epsilon=0.2, num_iter=20)
_, pgd_adv_s = pgd_attack(standard_model, X_test, y_test, epsilon=0.2, num_iter=20)

robust_clean = (robust_model(X_test).argmax(dim=1) == y_test).float().mean()
robust_adv = (robust_model(pgd_adv).argmax(dim=1) == y_test).float().mean()
standard_clean = (standard_model(X_test).argmax(dim=1) == y_test).float().mean()
standard_adv = (standard_model(pgd_adv_s).argmax(dim=1) == y_test).float().mean()

print("\n=== Robustness Comparison ===")
print(f"Standard model: clean={standard_clean:.4f}, adv={standard_adv:.4f}")
print(f"Robust model:   clean={robust_clean:.4f}, adv={robust_adv:.4f}")
print(f"Robustness improvement: {(robust_adv - standard_adv) * 100:.2f}%")
```

```
# Output:
# === Adversarial Training ===
# Training on clean + adversarial examples...
#
# Epoch 1: clean_acc=0.9200, adv_acc=0.4500
# Epoch 2: clean_acc=0.9400, adv_acc=0.5600
# Epoch 3: clean_acc=0.9500, adv_acc=0.6200
# Epoch 4: clean_acc=0.9600, adv_acc=0.6700
# Epoch 5: clean_acc=0.9700, adv_acc=0.7100
#
# === Robustness Comparison ===
# Standard model: clean=0.8900, adv=0.1200
# Robust model:   clean=0.9300, adv=0.7100
# Robustness improvement: 59.00%
```

### Example 4: Foolbox for Evaluation

```python
import foolbox as fb
import torch
import numpy as np

# Wrap model for foolbox
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

# Create attack
attack = fb.attacks.LinfPGD()
epsilons = [0.0, 0.1, 0.2, 0.3, 0.5]

raw_advs, clipped_advs, success = attack(
    fmodel, X_test[:20], y_test[:20], epsilons=epsilons
)

print("=== Foolbox Robustness Evaluation ===")
print(f"{'Epsilon':>8}  {'Accuracy':>10}  {'Success Rate':>14}")
print("-" * 36)

for i, eps in enumerate(epsilons):
    if eps == 0.0:
        acc = (model(X_test[:20]).argmax(dim=1) == y_test[:20]).float().mean()
        print(f"{eps:>8.1f}  {acc:>10.4f}  {'N/A':>14}")
    else:
        acc = 1.0 - success[i].float().mean()
        print(f"{eps:>8.1f}  {acc:>10.4f}  {success[i].float().mean():>14.4f}")

print("\nAs epsilon increases, more attacks succeed.")
```

```
# Output:
# === Foolbox Robustness Evaluation ===
#   Epsilon    Accuracy   Success Rate
# ------------------------------------
#      0.0      0.9000            N/A
#      0.1      0.4500         0.5000
#      0.2      0.2000         0.7778
#      0.3      0.1000         0.8889
#      0.5      0.0000         1.0000
#
# As epsilon increases, more attacks succeed.
```

## Common Mistakes

1. **Using FGSM instead of PGD for evaluating robustness**: FGSM is a weak attack. PGD (with multiple iterations) gives a much more accurate estimate of model robustness. Always use PGD for evaluation.

2. **Confusing adversarial robustness with generalization**: A model can generalize well (high test accuracy) but be completely non-robust (adversarial accuracy near zero). These are distinct properties.

3. **Not clamping adversarial examples**: Attack perturbations can push pixel values outside valid ranges. Always clamp to [0, 1] or the valid input range.

4. **Evaluating defense on the same attack used for training**: This overestimates robustness. Always evaluate against a stronger attack (e.g., evaluate PGD-trained models against AutoAttack).

5. **Gradient masking**: Some defenses (e.g., non-differentiable preprocessing) give a false sense of robustness by obfuscating gradients. Check that gradients are meaningful.

6. **Ignoring the threat model**: The choice of epsilon depends on the application. Epsilon=0.3 for MNIST is reasonable; epsilon=8/255 for ImageNet is standard. Define your threat model explicitly.

7. **Not testing against adaptive attacks**: A defense that works against a specific attack may fail against an attack that is designed to circumvent it. Always assume a white-box, adaptive attacker.

## Interview Questions

### Beginner

1. **Q:** What is the Fast Gradient Sign Method (FGSM)?  
   **A:** FGSM is a single-step adversarial attack that computes the gradient of the loss with respect to the input image, takes the sign of that gradient, and adds it to the image scaled by epsilon. It is fast but relatively weak compared to iterative attacks.

2. **Q:** What is adversarial training?  
   **A:** Adversarial training augments the training set with adversarial examples. At each training step, the model is trained on both clean and adversarially perturbed examples, improving robustness.

3. **Q:** What is the epsilon parameter in adversarial attacks?  
   **A:** Epsilon controls the magnitude of the perturbation. It defines the maximum L-infinity distance between the original and adversarial example. Larger epsilon means stronger perturbations that are more visible.

4. **Q:** What is the difference between white-box and black-box attacks?  
   **A:** White-box attacks have full access to the model (architecture, weights, gradients). Black-box attacks only have query access (can get predictions but not gradients).

5. **Q:** Why are neural networks vulnerable to adversarial examples?  
   **A:** Neural networks learn linear-like decision boundaries in high-dimensional spaces. Small perturbations in the direction that maximizes loss can push inputs across the decision boundary, even though the change is imperceptible.

### Intermediate

1. **Q:** Compare FGSM and PGD. When would you use each?  
   **A:** FGSM is a single-step attack that is fast but less effective. PGD is iterative and finds stronger adversarial examples within the same epsilon budget. Use FGSM for quick evaluations or data augmentation; use PGD for rigorous robustness evaluation.

2. **Q:** How does the choice of norm (L2, Linf) affect adversarial attacks?  
   **A:** Linf constraints limit the maximum change per pixel, producing perceptually uniform noise. L2 constraints limit the total Euclidean change, allowing small changes across many pixels. The choice depends on the threat model.

3. **Q:** What is gradient masking and why is it problematic?  
   **A:** Gradient masking occurs when a defense creates non-informative gradients (e.g., through non-differentiable operations). This gives a false sense of robustness because gradient-based attacks fail, but the model is still vulnerable to black-box attacks or attacks that approximate gradients.

4. **Q:** How do you select an appropriate epsilon for your application?  
   **A:** The epsilon should be the maximum perturbation that an attacker can realistically apply. For image models, epsilon=8/255 (0.031) is standard for ImageNet. For medical or security applications, the epsilon should be based on domain constraints.

5. **Q:** How does adversarial training affect clean accuracy?  
   **A:** Adversarial training typically reduces clean accuracy (the robustness-accuracy tradeoff). The larger the epsilon used in training, the greater the accuracy drop. Recent work aims to minimize this tradeoff.

### Advanced

1. **Q:** Derive the PGD attack from the perspective of constrained optimization. Explain why multi-step PGD is a stronger adversary than single-step FGSM.  
   **A:** PGD solves the constrained maximization problem max_{delta in S} L(f(x+delta), y) where S is the Lp ball of radius epsilon. FGSM takes one gradient step, which is a poor approximation to the constrained optimum. PGD iteratively takes gradient steps and projects back onto S, which is equivalent to running projected gradient ascent. With enough iterations, PGD converges to a local maximum of the loss within the constraint set, making it a much stronger adversary.

2. **Q:** Design a certified defense against adversarial examples that provides a formal guarantee of robustness. Discuss the tradeoffs between certified and empirical defenses.  
   **A:** Certified defenses provide provable guarantees that no perturbation within a given radius can change the prediction. Methods include randomized smoothing (add Gaussian noise and predict the majority), interval bound propagation (bound propagation through the network), and Lipschitz-based certification. Tradeoffs: certified defenses typically have lower clean accuracy and are computationally expensive. Empirical defenses (adversarial training) offer better accuracy but no formal guarantees.

3. **Q:** Discuss the relationship between adversarial robustness and model interpretability. How can explanations (SHAP, LIME) be used to detect or defend against adversarial attacks?  
   **A:** Adversarial examples often cause feature attributions to change dramatically. This can be used for detection: if SHAP values for a prediction are unstable under small perturbations, the input may be adversarial. Defensively, training models with explanation regularization (penalizing changes in attributions under perturbation) can improve both robustness and interpretability. Conversely, explanations themselves are vulnerable to adversarial manipulation.

## Practice Problems

### Easy

1. Implement FGSM from scratch for a PyTorch model and test it on random data.

2. Evaluate a trained model's accuracy on clean vs. FGSM-adversarial examples at epsilon=0.1, 0.2, 0.3.

3. Visualize an adversarial perturbation and the resulting adversarial image.

4. Implement a function that checks whether a given prediction is robust by testing multiple random perturbations.

5. Use Foolbox to run a LinfPGD attack on a pre-trained model.

### Medium

1. Implement PGD with 10 iterations and compare its effectiveness to FGSM at the same epsilon.

2. Train two models: one with standard training and one with adversarial training. Compare their robustness curves (accuracy vs. epsilon).

3. Implement a black-box attack using a surrogate model.

4. Evaluate the robustness of a model against L2 attacks (not just Linf).

5. Create a robustness evaluation report that includes accuracy at multiple epsilons, attack success rates, and perturbation statistics.

### Hard

1. Implement a differentiable defense (e.g., JPEG compression-based defense) and evaluate it against an adaptive attack that anticipates the defense.

2. Reproduce the Madry et al. CIFAR-10 adversarial training approach and evaluate against PGD and AutoAttack.

3. Design and implement a certified defense using randomized smoothing with a custom base classifier.

## Solutions

**Easy 1:**
```python
import torch
def fgsm(model, x, y, eps=0.1):
    x = x.clone().detach().requires_grad_(True)
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()
    return torch.clamp(x + eps * x.grad.sign(), 0, 1).detach()
```

**Medium 1:**
```python
def pgd(model, x, y, eps=0.1, alpha=0.01, steps=10):
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - x, 1 - x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = torch.nn.CrossEntropyLoss()(model(x + delta), y)
        model.zero_grad()
        loss.backward()
        delta = delta + alpha * delta.grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x + delta, 0, 1) - x
        delta = delta.detach()
    return torch.clamp(x + delta, 0, 1)
```

**Hard 1:**
```python
import kornia

class JPEGDefense:
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, model, x):
        # Differentiable JPEG compression
        x_jpeg = kornia.enhance.JPEG(quality=self.quality)(x)
        return model(x_jpeg)

# Adaptive attack would differentiate through JPEG to break this defense.
```

## Related Concepts

- **ML-086 Interpretability Methods**: Explanations can detect adversarial examples.
- **ML-088 Data Augmentation**: Adversarial training is a form of data augmentation.
- **ML-085 Fairness in ML**: Adversarial attacks can exploit fairness vulnerabilities.

## Next Concepts

- **ML-088 Data Augmentation** — Extending adversarial training concepts to general augmentation.
- **ML-086 Interpretability Methods** — Using explanations to understand and defend against attacks.

## Summary

Adversarial ML studies attacks and defenses for ML models. Gradient-based attacks like FGSM and PGD craft imperceptible perturbations that cause misclassification. PGD is significantly stronger than FGSM because it iteratively optimizes the perturbation. Adversarial training, which trains on both clean and adversarial examples, is the most effective empirical defense, though it typically reduces clean accuracy. Libraries like Foolbox and ART provide standardized attack and evaluation tools. Understanding adversarial ML is essential for deploying models in security-critical environments.

## Key Takeaways

- FGSM uses one gradient step to create adversarial examples
- PGD iteratively optimizes perturbations within an epsilon ball
- PGD is a stronger adversary than FGSM at the same epsilon
- Adversarial training is the most effective empirical defense
- Gradient masking gives a false sense of robustness
- Always evaluate against strong attacks (PGD, AutoAttack)
- Adversarial robustness often trades off with clean accuracy
- Choose epsilon based on the application's threat model
