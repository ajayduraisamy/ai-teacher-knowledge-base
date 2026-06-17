# Concept: Neural Networks

## Concept ID

DL-001

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Define artificial neural networks and explain their biological inspiration
- Identify the core components: neurons, weights, biases, activation functions, and layers
- Explain how depth enables hierarchical learning
- Implement a simple 2-layer neural network in PyTorch

## Prerequisites

- Basic Python programming
- Familiarity with functions and linear algebra (vectors, matrices, matrix multiplication)
- High school level calculus (derivatives)

## Definition

An artificial neural network (ANN) is a computational model inspired by the biological neural networks in animal brains. It consists of interconnected processing units called neurons, organized in layers, that transform input data through weighted connections and non-linear activation functions to produce outputs. Each connection between neurons has an associated weight that is adjusted during learning to minimize the error between predicted and target outputs.

Formally, a neural network defines a function $f: \mathbb{R}^n \to \mathbb{R}^m$ that maps an input vector $\mathbf{x} \in \mathbb{R}^n$ to an output vector $\mathbf{y} \in \mathbb{R}^m$. For a network with $L$ layers, this function is a composition of layer-wise transformations:

$$f(\mathbf{x}) = f_L(f_{L-1}(\dots f_1(\mathbf{x})\dots))$$

Each layer transformation $f_\ell$ typically takes the form:

$$\mathbf{h}_\ell = \sigma_\ell(\mathbf{W}_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell)$$

where $\mathbf{W}_\ell$ is the weight matrix, $\mathbf{b}_\ell$ is the bias vector, $\sigma_\ell$ is the activation function, and $\mathbf{h}_{\ell-1}$ is the output of the previous layer (with $\mathbf{h}_0 = \mathbf{x}$).

## Intuition

Think of a neural network as a multi-stage information processing pipeline. Raw data enters at the input layer and is progressively transformed as it passes through hidden layers. Each neuron in a hidden layer acts as a tiny feature detector: it receives signals from the previous layer, computes a weighted sum, applies a non-linear activation, and passes its output forward.

The "learning" process is akin to tuning the dials (weights) of a giant mixing console. Initially, the dials are set randomly and the output is garbage. Through iterative feedback (backpropagation), the dials are gradually adjusted so the network's output becomes increasingly accurate.

Depth — having multiple hidden layers — allows the network to build hierarchical representations. Early layers detect simple patterns (edges, textures in images; phonemes in audio), middle layers combine these into parts (object parts, syllables), and later layers assemble them into high-level concepts (objects, words). This hierarchical abstraction is what gives deep neural networks their power.

## Why This Concept Matters

Neural networks are the foundational building block of all modern deep learning. Understanding how neurons, weights, activations, and layers interact is essential for designing, debugging, and improving deep learning models. From image classification (CNNs) to language modeling (Transformers), every architecture builds upon these core principles. Mastery of neural networks enables practitioners to move beyond treating models as black boxes and instead develop intuition about why certain architectures work, how to tune them, and when they might fail.

## Real World Examples

1. **Image Recognition:** A neural network for object detection processes raw pixels through successive layers. The first layer detects edges and color gradients; deeper layers recognize shapes like wheels or windows; the final layer identifies objects like "car" or "person."

2. **Machine Translation:** A sequence-to-sequence neural network reads a sentence in English word by word, compresses its meaning into an internal representation, and then generates the equivalent sentence in French or German.

3. **Medical Diagnosis:** A neural network analyzes patient data (symptoms, lab results, medical images) and outputs probabilities for various conditions. The network learns complex, non-linear relationships between symptoms that would be difficult to capture with traditional statistical models.

## AI/ML Relevance

Neural networks form the backbone of virtually every modern AI system. They are the engine behind:

- **Computer Vision:** Convolutional Neural Networks (CNNs) are specialized neural nets for grid-like data.
- **Natural Language Processing:** Recurrent Neural Networks (RNNs) and Transformers handle sequential text data.
- **Reinforcement Learning:** Deep Q-Networks (DQNs) use neural nets as function approximators for Q-learning.
- **Generative Models:** Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are built from paired neural networks.

Any practitioner working in AI/ML must understand neural network fundamentals before progressing to specialized architectures.

## Mathematical Explanation

### Weighted Sum

Each neuron computes a weighted sum of its inputs plus a bias:

$$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

### Activation Function

The weighted sum is passed through a non-linear activation function $\sigma$:

$$a = \sigma(z)$$

Common activation functions:

- **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$, outputs in $(0, 1)$
- **ReLU:** $\sigma(z) = \max(0, z)$, most commonly used in hidden layers
- **Tanh:** $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, outputs in $(-1, 1)$

### Forward Propagation for a 2-Layer Network

Given input $\mathbf{x} \in \mathbb{R}^{d_{in}}$, hidden layer weights $\mathbf{W}_1 \in \mathbb{R}^{d_{hidden} \times d_{in}}$, bias $\mathbf{b}_1 \in \mathbb{R}^{d_{hidden}}$, output layer weights $\mathbf{W}_2 \in \mathbb{R}^{d_{out} \times d_{hidden}}$, bias $\mathbf{b}_2 \in \mathbb{R}^{d_{out}}$:

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
$$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$$
$$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$$
$$\mathbf{a}_2 = \text{Softmax}(\mathbf{z}_2)$$

The output $\mathbf{a}_2$ is a probability distribution over $d_{out}$ classes.

### Loss Function

For classification, cross-entropy loss measures the difference between predicted probabilities $\hat{\mathbf{y}} = \mathbf{a}_2$ and true labels $\mathbf{y}$:

$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{j=1}^{d_{out}} y_j \log(\hat{y}_j)$$

## Code Examples

### Example 1: Simple 2-Layer Neural Network in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model with 784 input features, 256 hidden units, 10 output classes
model = TwoLayerNet(784, 256, 10)
dummy_input = torch.randn(32, 784)  # batch of 32 samples
output = model(dummy_input)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([32, 10])
print(f"Model architecture:\n{model}")
# Output: Model architecture:
# TwoLayerNet(
#   (fc1): Linear(in_features=784, out_features=256, bias=True)
#   (fc2): Linear(in_features=256, out_features=10, bias=True)
# )
```

### Example 2: Manual Forward Pass with Weighted Sum

```python
import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Manual 2-layer forward pass
x = torch.tensor([0.5, -0.2, 0.1])  # input: 3 features

# Layer 1: 3 inputs -> 4 hidden neurons
W1 = torch.randn(4, 3) * 0.1
b1 = torch.zeros(4)
z1 = W1 @ x + b1
a1 = torch.relu(z1)
print(f"Hidden activations: {a1}")
# Output: Hidden activations: tensor([0.0000, 0.0143, 0.0248, 0.0000])

# Layer 2: 4 hidden -> 2 outputs
W2 = torch.randn(2, 4) * 0.1
b2 = torch.zeros(2)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
print(f"Output probabilities: {a2}")
# Output: Output probabilities: tensor([0.5021, 0.4989])
```

### Example 3: Training Loop (Forward, Loss, Backward)

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = TwoLayerNet(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data
X = torch.randn(64, 784)
y = torch.randint(0, 10, (64,))

# Training step
optimizer.zero_grad()
outputs = model(X)                        # forward pass
loss = criterion(outputs, y)             # compute loss
loss.backward()                          # backward pass
optimizer.step()                         # update weights

print(f"Loss after one step: {loss.item():.4f}")
# Output: Loss after one step: 2.3026

# Verify gradients exist
print(f"Gradients on fc1 weight: {model.fc1.weight.grad.norm():.6f}")
# Output: Gradients on fc1 weight: 2.457891
```

## Common Mistakes

1. **Using wrong output activation:** Applying sigmoid to the output layer for multi-class classification instead of softmax. Sigmoid treats each class independently; softmax ensures probabilities sum to 1.

2. **Forgetting to call `zero_grad()`:** Gradients accumulate by default in PyTorch. Skipping `optimizer.zero_grad()` causes gradients from previous steps to corrupt current updates.

3. **Initializing weights too large or too small:** Large initial weights cause saturated activations and vanishing gradients with sigmoid/tanh; small weights in deep networks cause activation collapse. Xavier/Glorot or He initialization are standard solutions.

4. **Using non-differentiable operations in the forward pass:** Operations like `torch.argmax` or `torch.round` break gradient flow. Always use differentiable surrogates during training (e.g., softmax instead of argmax).

5. **Misinterpreting logits vs probabilities:** `CrossEntropyLoss` in PyTorch expects raw logits (not softmax outputs). Applying softmax before the loss function leads to double-softmax and incorrect gradients.

## Interview Questions

### Beginner

1. What is a neural network and what are its core components?
2. Explain the role of weights and biases in a neural network.
3. What is the purpose of a non-linear activation function? What happens if you remove it?
4. How does information flow from input to output in a feedforward neural network?
5. What is a "layer" in a neural network? What distinguishes input, hidden, and output layers?

### Intermediate

1. Why is depth important in neural networks? What advantages do deeper networks have over shallow ones?
2. Explain the vanishing gradient problem. Which activation functions help mitigate it and why?
3. How does the choice of activation function affect training dynamics and convergence?
4. Compare and contrast weight initialization strategies (Xavier vs He). When would you use each?
5. What is the difference between a neural network with one hidden layer and a deep neural network with many layers in terms of representational power?

### Advanced

1. Derive the gradient of the cross-entropy loss with respect to the output layer weights for a 2-layer network.
2. Explain the concept of the "neural tangent kernel" and how it relates to infinite-width neural networks.
3. Discuss the lottery ticket hypothesis and its implications for neural network pruning and training.

## Practice Problems

### Easy

1. Create a neural network in PyTorch with 1 hidden layer of 128 neurons for a 20-class classification problem with 100 input features.
2. Write a function that computes the forward pass of a single neuron given an input vector, weight vector, and bias.
3. Calculate the number of parameters in a network with input 784, hidden 512, and 10 outputs (including biases).
4. Modify the `TwoLayerNet` class to use tanh activation instead of ReLU.
5. Create a 3-layer network (input 64, hidden1 128, hidden2 64, output 10) in PyTorch.

### Medium

1. Implement a custom linear layer from scratch using only `torch.Tensor` and autograd. Verify it produces the same output as `nn.Linear`.
2. Train a 2-layer network on the MNIST dataset. Track training and validation loss for 10 epochs.
3. Experiment with different activation functions (ReLU, Tanh, Sigmoid) on a simple 2D binary classification problem. Compare convergence speed and decision boundaries.
4. Implement weight decay (L2 regularization) manually within a training loop and compare results to PyTorch's built-in `weight_decay` parameter.
5. Write a learning rate scheduler that reduces the learning rate by half whenever validation loss plateaus for 3 epochs.

### Hard

1. Implement forward and backward passes for a 3-layer network entirely manually (without autograd) using NumPy. Verify gradients against autograd.
2. Design and train a neural network that can solve the XOR problem using a single hidden layer. Visualize the decision boundary at different stages of training.
3. Implement a residual block and show how it improves gradient flow in a 10-layer network compared to a plain network.

## Solutions

### Easy 1
```python
model = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 20)
)
# Parameters: 100*128 + 128 + 128*20 + 20 = 15,628
```

### Easy 3
```python
# Layer 1: 784*512 + 512 = 401,920
# Layer 2: 512*10 + 10 = 5,130
# Total: 407,050 parameters
```

### Medium 1
```python
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias
```

### Hard 1
```python
import numpy as np

def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)
    z2 = W2 @ a1 + b2
    a2 = 1 / (1 + np.exp(-z2))
    cache = (x, z1, a1, z2, a2)
    return a2, cache

def backward(y, cache, W2):
    x, z1, a1, z2, a2 = cache
    dz2 = a2 - y
    dW2 = np.outer(dz2, a1)
    db2 = dz2
    da1 = W2.T @ dz2
    dz1 = da1 * (z1 > 0)
    dW1 = np.outer(dz1, x)
    db1 = dz1
    return dW1, db1, dW2, db2
```

## Related Concepts

- Activation Functions (sigmoid, tanh, ReLU)
- Backpropagation
- Loss Functions (cross-entropy, MSE)
- Weight Initialization
- Gradient Descent

## Next Concepts

- Convolutional Neural Networks
- Recurrent Neural Networks
- Batch Normalization
- Dropout
- Residual Networks

## Summary

Neural networks are computational models inspired by biological neurons, consisting of layers of interconnected processing units. Each neuron computes a weighted sum of its inputs followed by a non-linear activation. The network learns by adjusting weights to minimize a loss function through forward and backward propagation. Depth enables hierarchical feature learning, making deep networks powerful tools for complex pattern recognition tasks.

## Key Takeaways

- Neural networks consist of neurons organized in layers with weighted connections
- Activation functions introduce non-linearity, enabling the network to learn complex patterns
- Forward pass: input propagates through layers, each applying linear transform + activation
- A 2-layer network can already approximate a wide range of functions given sufficient hidden units
- Modern deep learning builds on these fundamentals with specialized architectures for different data modalities
