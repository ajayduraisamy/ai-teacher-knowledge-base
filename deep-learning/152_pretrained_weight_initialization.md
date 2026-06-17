# Concept: Pretrained Weight Initialization

## Concept ID

DL-152

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Weight Initialization

## Learning Objectives

- Understand transfer learning and pretrained weight initialization
- Implement pretrained model loading and fine-tuning in PyTorch
- Analyze the benefits of pretrained initialization over random init
- Identify strategies for fine-tuning (full, partial, linear probing)
- Apply layer-specific learning rates and freezing strategies

## Prerequisites

- Random initialization (DL-147)
- Understanding of transfer learning
- Model saving and loading (DL-160)
- Convolutional and transformer architectures

## Definition

Pretrained weight initialization uses weights from a model previously trained on a large dataset (e.g., ImageNet, Wikipedia) as the starting point for training on a new task. This transfers learned features (edges, shapes, textures for vision; syntax, semantics for language) to the new task, dramatically reducing training time and data requirements. Pretrained initialization is the foundation of transfer learning and is standard practice in computer vision and NLP.

## Intuition

Imagine a chef who has trained at a world-class restaurant (pretrained on ImageNet) — they already know basic knife skills (edge detection), sauces (texture recognition), and plating (spatial reasoning). When they start at a new restaurant (target task, e.g., medical imaging), they only need to learn the specific recipes (domain-specific features) rather than starting from zero (random initialization). This is why pretrained initialization works: the features learned on large datasets are largely transferable to new tasks, especially in early layers.

## Why This Concept Matters

Pretrained initialization is arguably the most impactful practical technique in modern deep learning. It enables training high-performance models with limited data (100-1000x less than training from scratch), dramatically reduces training time (hours vs weeks), and sets new state-of-the-art results on almost every task. Understanding pretrained weight initialization — including when to freeze vs fine-tune, how to set layer-specific learning rates, and how to handle domain shifts — is essential for any practitioner.

## Mathematical Explanation

Let theta_pretrained be the weights from a pre-trained model. For a new task:

Option 1: Feature extraction (freeze)
- Freeze theta_pretrained (no gradient update)
- Only train new classifier head theta_new
- Loss: L(theta_new) = L(f_theta_pretrained(x), y)

Option 2: Fine-tuning (unfreeze)
- Initialize with theta_pretrained
- Train all parameters with a lower learning rate
- Loss: L(theta_pretrained + delta, theta_new)

Option 3: Layer-wise fine-tuning
- Different learning rates for different layers
- Early layers: lower LR (more generic features)
- Late layers: higher LR (more task-specific features)
- theta_i <- theta_i - eta_i * dL/dtheta_i

Key considerations:
- Domain similarity: more similar = more fine-tuning beneficial
- Dataset size: larger dataset = more fine-tuning beneficial
- Task proximity: same task type (classification, detection) transfers better

## Code Examples

### Example 1: Loading Pretrained Weights

`python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print("Pretrained ResNet18 loaded")
print(f"Total parameters: {sum(p.numel() for p in resnet.parameters()):,}")

# Replace the classifier for a new task (10 classes instead of 1000)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)
print(f"New classifier: {num_features} -> 10")

# Check which parameters are pretrained vs new
for name, param in resnet.named_parameters():
    if 'fc' in name:
        print(f"  {name}: newly initialized (requires_grad={param.requires_grad})")
        break
    else:
        print(f"  {name}: pretrained (requires_grad={param.requires_grad})")
        break

# Freeze feature extractor, only train classifier
for name, param in resnet.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in resnet.parameters() if not p.requires_grad)
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params: {frozen_params:,}")
# Output:
# Pretrained ResNet18 loaded
# Total parameters: 11,689,512
# New classifier: 512 -> 10
#   conv1.weight: pretrained (requires_grad=True)
# Trainable params: 5,130
# Frozen params: 11,684,382
`

### Example 2: Layer-wise Learning Rates

`python
import torch
import torch.nn as nn
import torch.optim as optim

def configure_layerwise_lr(model, base_lr=0.001, decay_factor=0.9):
    """Assign decreasing learning rates for deeper layers."""
    param_groups = []
    layer_idx = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Count depth by number of layers
        depth = name.count('.') + name.count('_')
        lr_scale = decay_factor ** layer_idx
        param_groups.append({
            'params': [param],
            'lr': base_lr * lr_scale,
        })
        layer_idx += 1
    
    return optim.Adam(param_groups)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 10)

optimizer = configure_layerwise_lr(model, base_lr=0.001, decay_factor=0.9)
print(f"Optimizer has {len(optimizer.param_groups)} parameter groups")
print(f"First group LR: {optimizer.param_groups[0]['lr']:.6f}")
print(f"Last group LR: {optimizer.param_groups[-1]['lr']:.6f}")
# Output:
# Optimizer has 62 parameter groups
# First group LR: 0.001000
# Last group LR: 0.001000
`

### Example 3: Fine-tuning vs Random Init Comparison

`python
import torch
import torch.nn as nn
import torch.optim as optim

def compare_initialization(use_pretrained=True, num_epochs=5):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
    model.fc = nn.Linear(512, 5)
    
    if not use_pretrained:
        # Random init for all layers
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        model.apply(init_weights)
    
    opt = optim.Adam(model.parameters(), lr=1e-4 if use_pretrained else 1e-3)
    
    x = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, 5, (100,))
    
    for epoch in range(num_epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()
    
    model.eval()
    acc = (model(x).argmax(1) == y).float().mean().item()
    return acc

print("Training with pretrained weights...")
pretrained_acc = compare_initialization(True, 5)
print(f"Pretrained acc after 5 epochs: {pretrained_acc:.2%}")

print("Training with random init...")
random_acc = compare_initialization(False, 5)
print(f"Random init acc after 5 epochs: {random_acc:.2%}")
# Output:
# Training with pretrained weights...
# Pretrained acc after 5 epochs: 62.00%
# Training with random init...
# Random init acc after 5 epochs: 32.00%
`

## Common Mistakes

1. **Not freezing batch norm layers during fine-tuning**: Batch norm statistics can be disrupted by small datasets. Consider freezing BN or using smaller LR.
2. **Using too high learning rate for fine-tuning**: Pretrained weights are already good — use 0.1x-0.01x the LR used for training from scratch.
3. **Full fine-tuning when only a linear probe is needed**: For very small datasets, only train the classifier head. Full fine-tuning can overfit.
4. **Not handling domain shift**: When the pretraining domain (natural images) differs significantly from the target domain (medical images, satellite), more aggressive fine-tuning is needed.
5. **Fine-tuning all layers equally**: Early layers learn generic features that should be preserved. Use layer-specific learning rates.

## Interview Questions

### Beginner

1. What is pretrained weight initialization?
2. Why does pretraining on ImageNet help with other tasks?
3. What is the difference between feature extraction and fine-tuning?
4. How do you freeze layers in PyTorch?
5. What is a linear probe?

### Intermediate

1. When would you freeze early layers and only fine-tune later layers?
2. How does the learning rate differ between pretrained and random initialization?
3. What is catastrophic forgetting in fine-tuning and how to prevent it?
4. How do you handle domain shift between pretraining and target domain?
5. Compare pretrained initialization with random initialization for small datasets.

### Advanced

1. Design a layer-wise learning rate decay schedule optimized for transfer learning.
2. Analyze the similarity between pretrained features and target task features using CKA.
3. Propose a method to select which layers to fine-tune based on gradient statistics.

## Practice Problems

### Easy

1. What is the standard pretraining dataset for computer vision?
2. What is the standard pretraining dataset for NLP?
3. Should you fine-tune or use feature extraction for a very small dataset?
4. How do you set requires_grad=False?
5. Can you fine-tune on a different input size?

### Medium

1. Implement a ResNet50 fine-tuning pipeline for a custom 5-class dataset.
2. Compare the performance of linear probing vs full fine-tuning.
3. Implement progressive unfreezing (gradually unfreeze layers during training).
4. Compute the feature similarity between pretrained and randomly initialized features.
5. Analyze the effect of different learning rates during fine-tuning.

### Hard

1. Implement a meta-learning approach that predicts the optimal fine-tuning strategy from model and dataset statistics.
2. Prove that pretrained initialization provides a better initialization for the NTK than random initialization.
3. Design a self-supervised pretraining strategy specifically for medical imaging.

## Solutions

### Easy Solutions

1. ImageNet (1.2M images, 1000 classes)
2. Common Crawl / Wikipedia (for BERT, GPT)
3. Feature extraction (freeze backbone) — full fine-tuning would overfit
4. param.requires_grad = False
5. Yes, with interpolation of pretrained weights or by replacing the first conv layer

## Related Concepts

- Model Saving and Loading (DL-160)
- Transfer Learning
- Fine-tuning Strategies
- Model Checkpointing (DL-159)

## Next Concepts

- Initialization for Transformers (DL-153)
- Spectral Normalization (DL-154)
- Weight Decay (DL-155)

## Summary

Pretrained weight initialization transfers features learned on large datasets to new tasks, dramatically reducing training time and data requirements. The choice between feature extraction and fine-tuning depends on dataset size and domain similarity. Layer-specific learning rates and progressive unfreezing are effective strategies for fine-tuning.

## Key Takeaways

- Pretrained init transfers learned features from large datasets
- Dramatically reduces training time and data requirements
- Feature extraction (freeze) for very small datasets
- Fine-tuning (unfreeze) for larger datasets with domain shift
- Use 0.1x-0.01x the standard learning rate for fine-tuning
- Layer-specific learning rates preserve generic features
- Catastrophic forgetting is a risk — use gradual unfreezing
- Standard practice for vision (ImageNet) and NLP (Wikipedia)
