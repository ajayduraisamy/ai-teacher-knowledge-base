# Concept: PyTorch Neural Networks (nn.Module)

## Concept ID

PYT-097

## Difficulty

Advanced

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Build neural network models using `torch.nn.Module` and `nn.Sequential`
- Implement linear layers, convolutional layers, activation functions, and dropout
- Use loss functions (`nn.CrossEntropyLoss`, `nn.MSELoss`) and optimizers (`torch.optim.Adam`, `torch.optim.SGD`)
- Create custom modules with a forward method

## Prerequisites

- PYT-096 — PyTorch Tensors (tensor operations, autograd)
- Basic understanding of neural network architecture (layers, activation, loss)
- Familiarity with supervised learning concepts

## Definition

PyTorch's `torch.nn` module provides building blocks for constructing neural networks:

**Core Components:**

- **`nn.Module`:** Base class for all neural network modules. Models inherit from it and define layers in `__init__()` and the forward pass in `forward()`.
- **`nn.Sequential`:** Ordered container for sequential application of layers. No need to write forward() explicitly.
- **`nn.Linear(in_features, out_features)`:** Fully connected (dense) layer with learnable weights and bias.
- **`nn.Conv2d(in_channels, out_channels, kernel_size)`:** 2D convolutional layer.
- **`nn.ReLU()` / `nn.Sigmoid()` / `nn.Tanh()`:** Activation functions.
- **`nn.Dropout(p=0.5)`:** Regularization layer that randomly zeros out neurons during training.
- **`nn.MaxPool2d(kernel_size)`:** Downsampling for convolutional networks.
- **`nn.BatchNorm1d` / `nn.BatchNorm2d`:** Normalize activations to stabilize training.

**Loss Functions:**

- `nn.CrossEntropyLoss()`: Combines LogSoftmax + NLLLoss for multi-class classification
- `nn.MSELoss()`: Mean squared error for regression
- `nn.BCEWithLogitsLoss()`: Binary classification with sigmoid built-in
- `nn.L1Loss()`: Mean absolute error

**Optimizers (`torch.optim`):**

- `optim.Adam(params, lr)`: Adaptive Moment Estimation — most popular default
- `optim.SGD(params, lr, momentum)`: Stochastic gradient descent with momentum
- `optim.RMSprop(params, lr)`: Root mean square propagation

## Intuition

Think of `nn.Module` as a LEGO building block system. You define the architecture by composing small blocks (Linear → ReLU → Dropout → Linear) into larger blocks (a layer → a block → the full model). Each block can compute its own forward pass, and PyTorch automatically tracks parameters for optimization.

`nn.Sequential` is like a recipe: you list the layers in order, and data flows through them sequentially. For more complex architectures (multiple inputs, branching, skip connections), you write a custom `nn.Module` with an explicit `forward()` method.

## Why This Concept Matters

- **Rapid Prototyping:** nn.Sequential lets you build and test architectures in minutes
- **Modularity:** Custom modules enable reusable components (residual blocks, attention heads)
- **Automatic Parameter Tracking:** `model.parameters()` returns all learnable weights — no manual bookkeeping
- **Production Ready:** nn.Module is the standard for all PyTorch models, from research prototypes to production inference
- **Transfer Learning:** Load pretrained models and fine-tune by freezing/unfreezing specific modules

## Real World Examples

1. **Image Classification:** A ResNet-50 model (torchvision) built from nn.Conv2d, nn.BatchNorm2d, nn.ReLU, and residual connections via custom nn.Module.
2. **Sentiment Analysis:** An LSTM-based text classifier using nn.LSTM, nn.Embedding, nn.Dropout, and nn.Linear.
3. **Regression:** A 3-layer MLP (multilayer perceptron) with ReLU activations predicting house prices using nn.Sequential.
4. **Generative Models:** A Variational Autoencoder (VAE) with custom encoder/decoder modules using nn.Conv2d and nn.ConvTranspose2d.
5. **Reinforcement Learning:** A policy network (nn.Module) that maps state vectors to action probabilities, trained with policy gradients.

## AI/ML Relevance

- **Deep Learning Foundation:** PyTorch nn.Module is the standard for all modern deep learning
- **Industry Adoption:** PyTorch is the most popular research framework; nn.Module is its core
- **Flexibility:** Custom forward passes enable novel architectures (Transformers, Graph Neural Nets)
- **Transfer Learning:** nn.Module's `load_state_dict` enables loading pretrained weights
- **HuggingFace Integration:** Transformers library uses nn.Module for all models

## Code Examples

### Example 1: Simple linear regression model (nn.Module)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
print(f"Model: {model}")
print(f"Parameters: {list(model.parameters())}")

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.5

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

weight = model.linear.weight.item()
bias = model.linear.bias.item()
print(f"Learned: y = {weight:.3f}x + {bias:.3f} (true: 3x + 2)")
```
```
# Output:
# Model: LinearRegression(
#   (linear): Linear(in_features=1, out_features=1, bias=True)
# )
# Parameters: [Parameter containing: tensor([[-0.4978]]), Parameter containing: tensor([0.3521])]
# Epoch 20, Loss: 2.4123
# Epoch 40, Loss: 0.7789
# Epoch 60, Loss: 0.3632
# Epoch 80, Loss: 0.2766
# Epoch 100, Loss: 0.2643
# Learned: y = 2.841x + 2.103 (true: 3x + 2)
```

### Example 2: MLP for classification using nn.Sequential
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

model = MLPClassifier(input_dim=20, hidden_dim=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            _, predicted = torch.max(model(X_test), 1)
            accuracy = (predicted == y_test).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Acc: {accuracy:.3f}")
```
```
# Output:
# Epoch 10, Loss: 0.8745, Test Acc: 0.785
# Epoch 20, Loss: 0.5643, Test Acc: 0.825
# Epoch 30, Loss: 0.4231, Test Acc: 0.840
# Epoch 40, Loss: 0.3412, Test Acc: 0.845
# Epoch 50, Loss: 0.2876, Test Acc: 0.840
```

### Example 3: Convolutional neural network (CNN)
```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 28/2/2 = 7 for MNIST
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

model = CNN(num_classes=10)
print(f"CNN architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# Forward a fake image batch
dummy_input = torch.randn(4, 1, 28, 28)  # batch=4, 1 channel, 28x28
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # (4, 10)
```
```
# Output:
# CNN architecture:
# CNN(
#   (conv_layers): Sequential(
#     (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): ReLU()
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Flatten()
#     (1): Linear(in_features=3136, out_features=128, bias=True)
#     (2): ReLU()
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=128, out_features=10, bias=True)
#   )
# )
# Total params: 418,442
# Trainable params: 418,442
# Output shape: torch.Size([4, 10])
```

### Example 4: Custom module with skip connection (ResNet block)
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Skip connection: project input to output dimension if needed
        self.shortcut = nn.Identity() if in_features == out_features \
                       else nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = out + residual  # skip connection
        out = self.relu(out)
        return out

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.block1 = ResidualBlock(64, 64)
        self.block2 = ResidualBlock(64, 64)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.output_layer(x)
        return x

model = ResidualMLP(input_dim=10, num_classes=2)
dummy = torch.randn(8, 10)
print(f"Residual MLP output shape: {model(dummy).shape}")
print(f"Model:\n{model}")
```
```
# Output:
# Residual MLP output shape: torch.Size([8, 2])
# Model:
# ResidualMLP(
#   (input_layer): Linear(in_features=10, out_features=64, bias=True)
#   (block1): ResidualBlock(...)
#   (block2): ResidualBlock(...)
#   (output_layer): Linear(in_features=64, out_features=2, bias=True)
# )
```

### Example 5: Training loop with model.eval() and torch.no_grad()
```python
model = MLPClassifier(input_dim=20, hidden_dim=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_accs = []

for epoch in range(30):
    # Training
    model.train()
    optimizer.zero_grad()
    train_out = model(X_train)
    train_loss = criterion(train_out, y_train)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        _, predicted = torch.max(test_out, 1)
        accuracy = (predicted == y_test).float().mean()
        test_accs.append(accuracy.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss={train_loss.item():.4f}, Test Acc={accuracy:.3f}")

print(f"Final test accuracy: {test_accs[-1]:.3f}")
```
```
# Output:
# Epoch 10: Train Loss=0.4912, Test Acc=0.835
# Epoch 20: Train Loss=0.2978, Test Acc=0.850
# Epoch 30: Train Loss=0.2214, Test Acc=0.855
# Final test accuracy: 0.855
```

### Example 6: Transfer learning — freezing pretrained layers
```python
import torchvision.models as models

# Load pretrained ResNet18
resnet = models.resnet18(pretrained=True)
print(f"Original FC layer: {resnet.fc}")

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the final classifier for a new task (10 classes)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)

# Only the new fc layer is trainable
trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
total = sum(p.numel() for p in resnet.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# Verify freezing
for name, param in resnet.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
```
```
# Output:
# Original FC layer: Linear(in_features=512, out_features=1000, bias=True)
# Trainable params: 5,130 / 11,835,562
# Trainable: fc.weight
# Trainable: fc.bias
```

### Example 7: Multiple loss functions
```python
class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 32)
        self.classifier = nn.Linear(32, 2)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        features = torch.relu(self.shared(x))
        return self.classifier(features), self.regressor(features)

model = MultiOutputModel()
x = torch.randn(16, 10)
y_class = torch.randint(0, 2, (16,))
y_reg = torch.randn(16, 1)

out_class, out_reg = model(x)
loss_class = nn.CrossEntropyLoss()(out_class, y_class)
loss_reg = nn.MSELoss()(out_reg, y_reg)
total_loss = loss_class + 0.5 * loss_reg  # weighted sum
print(f"Classification loss: {loss_class:.3f}")
print(f"Regression loss: {loss_reg:.3f}")
print(f"Total loss: {total_loss:.3f}")
```
```
# Output:
# Classification loss: 0.721
# Regression loss: 1.234
# Total loss: 1.338
```

### Example 8: Weight initialization
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        print(f"  Initialized {m}")

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
model.apply(init_weights)

# Also access individual layer weights
layer = model[0]
print(f"\nWeight stats: mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
print(f"Bias stats: mean={layer.bias.mean():.4f}")
```
```
# Output:
#   Initialized Linear(in_features=10, out_features=32, bias=True)
#   Initialized Linear(in_features=32, out_features=1, bias=True)
# Weight stats: mean=0.0012, std=0.3156
# Bias stats: mean=0.0000
```

### Example 9: Model saving and loading
```python
model = MLPClassifier(input_dim=20, hidden_dim=64, num_classes=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train briefly (example)
model.train()
for _ in range(5):
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(X_train), y_train)
    loss.backward()
    optimizer.step()

# Save checkpoint
checkpoint = {
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}
torch.save(checkpoint, 'checkpoint.pth')
print(f"Model saved. Loss: {loss.item():.4f}")

# Load checkpoint
model2 = MLPClassifier(input_dim=20, hidden_dim=64, num_classes=3)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
checkpoint = torch.load('checkpoint.pth')
model2.load_state_dict(checkpoint['model_state_dict'])
optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
print(f"Model loaded from epoch {checkpoint['epoch']}")

# Verify
model2.eval()
with torch.no_grad():
    out = model2(X_test[:4])
    print(f"Loaded model predictions shape: {out.shape}")
```
```
# Output:
# Model saved. Loss: 1.3421
# Model loaded from epoch 5
# Loaded model predictions shape: torch.Size([4, 3])
```

### Example 10: Using GPU with nn.Module
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MLPClassifier(input_dim=20, hidden_dim=128, num_classes=3).to(device)
X_train_device = X_train.to(device)
y_train_device = y_train.to(device)
X_test_device = X_test.to(device)
y_test_device = y_test.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train_device)
    loss = criterion(outputs, y_train_device)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            _, pred = torch.max(model(X_test_device), 1)
            acc = (pred == y_test_device).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc:.3f}")
```
```
# Output:
# Using device: cpu
# Epoch 5, Loss: 0.7341, Acc: 0.800
# Epoch 10, Loss: 0.5321, Acc: 0.830
```

## Common Mistakes

1. **Calling `model()` instead of `model.forward()`.** They are the same because `__call__` invokes `forward`, but `__call__` also handles hooks. Always use `model(x)`, never `model.forward(x)`.
2. **Forgetting `model.train()` and `model.eval()`.** Dropout and BatchNorm behave differently for training vs evaluation. Failing to toggle `model.eval()` before inference causes incorrect results.
3. **Not calling `optimizer.zero_grad()` before each backward pass.** Gradients accumulate by default. Missing `zero_grad()` doubles the gradient each iteration.
4. **Moving model to device but not input tensors.** `model.to(device)` moves parameters but input tensors must be moved separately: `input = input.to(device)`.
5. **Using `CrossEntropyLoss` with one-hot targets.** `CrossEntropyLoss` expects class indices (long tensor of shape [N]), not one-hot vectors. Use `MSELoss` or `BCEWithLogitsLoss` for one-hot targets.
6. **Misunderstanding the `torch.max(outputs, 1)` call.** `torch.max` returns both values and indices. Use `_, predicted = torch.max(outputs, 1)` to get the class prediction.
7. **Calling `backward()` on a non-scalar loss without gradient parameter.** `loss` must be a scalar. Most loss functions return a scalar, but if you sum manually, ensure it's a scalar.

## Interview Questions

### Beginner - 5

1. **Q:** What is the base class for all PyTorch models?  
   **A:** `nn.Module`. Custom models inherit from it and override `__init__` and `forward`.

2. **Q:** What is `nn.Sequential` used for?  
   **A:** It's a container that applies layers in sequential order. Data flows through each layer in the order they were registered.

3. **Q:** How do you move a model to GPU?  
   **A:** `model = model.to('cuda')` or `model.cuda()`. All parameters and buffers are moved.

4. **Q:** What does `model.eval()` do?  
   **A:** It sets the model to evaluation mode, disabling dropout and fixing batch normalization statistics. Always call it before inference.

5. **Q:** How do you access all trainable parameters of a model?  
   **A:** `model.parameters()` returns an iterator over all parameters. `model.named_parameters()` returns name-parameter pairs.

### Intermediate - 5

1. **Q:** What is the difference between `nn.CrossEntropyLoss` and `nn.NLLLoss`?  
   **A:** `CrossEntropyLoss` combines `LogSoftmax` + `NLLLoss` in one class. `NLLLoss` expects log-probabilities (output of `LogSoftmax`) as input. For most cases, use `CrossEntropyLoss` directly.

2. **Q:** How do you freeze layers in a pretrained model?  
   **A:** Set `param.requires_grad = False` for the layers to freeze, then only pass parameters with `requires_grad=True` to the optimizer.

3. **Q:** Explain the purpose of `torch.no_grad()` during evaluation.  
   **A:** It disables gradient tracking, reducing memory usage and speeding up computation. Gradients are not needed during inference, so the computational graph is not built.

4. **Q:** How does `nn.Dropout` work differently in train vs eval mode?  
   **A:** During training, dropout randomly zeros a fraction of neurons (specified by `p`). During eval, all neurons are active, and outputs are scaled by `1-p` to maintain expected magnitude.

5. **Q:** What is the `state_dict` in PyTorch models?  
   **A:** `state_dict` is a Python dictionary mapping each layer name to its learnable parameters (weights and biases). Used for saving/loading model checkpoints.

### Advanced - 3

1. **Q:** Explain how hooks (`register_forward_hook`, `register_backward_hook`) work in nn.Module and when you'd use them.  
   **A:** Hooks allow you to inspect or modify activations during forward/backward passes without modifying the model code. Used for feature visualization, gradient clipping, or debugging intermediate layers.

2. **Q:** How would you implement a custom autograd Function and integrate it into an nn.Module?  
   **A:** Subclass `torch.autograd.Function`, define `forward(ctx, ...)` and `backward(ctx, grad_output)`. Wrap it as a callable and use it inside the module's `forward` method: `CustomFunction.apply(x)`.

3. **Q:** Describe how PyTorch's distributed training (DDP) handles model parameters across multiple GPUs.  
   **A:** `DistributedDataParallel` wraps the model, synchronizes gradients across processes during backward, and ensures all replicas have the same parameters after each step. Each process handles a subset of the batch.

## Practice Problems

### Easy - 5

1. **E1:** Build a 2-layer MLP (Linear → ReLU → Linear) using nn.Sequential for binary classification.
2. **E2:** Count the total and trainable parameters of a model with 3 linear layers: 10→32→16→2.
3. **E3:** Create a CNN with one Conv2d(1, 16, 3), ReLU, MaxPool2d(2), Flatten, Linear(16*13*13, 10).
4. **E4:** Train a LinearRegression model (nn.Linear(1,1)) on y = 5x + 3 with noise.
5. **E5:** Save and load a model's state_dict.

### Medium - 5

1. **M1:** Build an MLP for the Iris dataset (4 features, 3 classes) with one hidden layer of 16 units. Train to >95% accuracy.
2. **M2:** Add dropout (p=0.5) and batch normalization to the MLP from M1. Compare training curves.
3. **M3:** Implement a custom ResidualBlock as an nn.Module and build a 3-block residual network for classification.
4. **M4:** Use a pretrained ResNet18 from torchvision, freeze all layers except the final fc, and train on a 5-class dataset.
5. **M5:** Implement a multi-task model with shared layers and two output heads (classification + regression).

### Hard - 3

1. **H1:** Implement a Variational Autoencoder (VAE) with nn.Module, including reparameterization trick in forward().
2. **H2:** Build a custom module that implements a Transformer encoder block (self-attention + FFN) without using nn.Transformer.
3. **H3:** Implement a Siamese network with shared weights (tie two subnetworks) using nn.Module and compute contrastive loss.

## Solutions

### E1 Solution
```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
```

### E2 Solution
```python
model = nn.Sequential(nn.Linear(10, 32), nn.Linear(32, 16), nn.Linear(16, 2))
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total}, Trainable: {trainable}")
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions extend the examples with training loops and custom architectures.

### H1-H3 Solutions require advanced PyTorch features and are beyond brief code blocks.

## Related Concepts

- 096 — PyTorch Tensors (foundation for nn.Module operations)
- 099 — Training Loops (full training workflow with nn.Module)
- 098 — TensorFlow/Keras (equivalent model building in TensorFlow)

## Next Concepts

- 099 — Training Loops (custom training loops, checkpointing, early stopping)
- 098 — TensorFlow/Keras (alternative deep learning framework)
- 100 — Project Structure (organizing deep learning projects)

## Summary

PyTorch's `nn.Module` is the base class for building neural networks. It provides automatic parameter tracking, GPU device management, and integration with autograd. `nn.Sequential` enables quick layer stacking; custom modules with explicit `forward()` enable complex architectures. Loss functions (`nn.CrossEntropyLoss`, `nn.MSELoss`) and optimizers (`optim.Adam`, `optim.SGD`) complete the training setup.

## Key Takeaways

- All models inherit from `nn.Module` and define `__init__` + `forward`
- `nn.Sequential` for simple feed-forward architectures
- Always toggle `model.train()` / `model.eval()` for correct dropout/batchnorm behavior
- Use `torch.no_grad()` during evaluation to save memory
- `model.state_dict()` contains all parameters for saving/loading
- Move both model and data to the same device with `.to(device)`
- Freeze pretrained layers with `requires_grad = False`
- `optimizer.zero_grad()` before each backward pass is mandatory
