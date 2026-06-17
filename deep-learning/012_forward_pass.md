# Concept: Forward Pass

## Concept ID

DL-012

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Define the forward pass in a neural network
- Explain the matrix multiplication view of forward propagation
- Implement forward passes with different activation functions
- Trace data shape transformations through network layers

## Prerequisites

- Neural Networks (DL-001)
- Linear algebra (matrix multiplication)
- Basic Python with PyTorch

## Definition

The forward pass (also called forward propagation) is the process of computing a neural network's output from its input by sequentially applying each layer's transformation. Given an input $\mathbf{x}$, the forward pass through an $L$-layer network computes:

$$\mathbf{a}^{(0)} = \mathbf{x}$$
$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)} \quad \text{for } \ell = 1, \dots, L$$
$$\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)}) \quad \text{for } \ell = 1, \dots, L$$
$$\hat{\mathbf{y}} = \mathbf{a}^{(L)}$$

where $\mathbf{W}^{(\ell)}$, $\mathbf{b}^{(\ell)}$ are the weights and biases of layer $\ell$, $\sigma^{(\ell)}$ is the activation function, $\mathbf{z}^{(\ell)}$ is the pre-activation (logits), and $\mathbf{a}^{(\ell)}$ is the post-activation (hidden representation).

## Intuition

Think of the forward pass as an information pipeline. Raw data enters at one end and is progressively transformed, filtered, and refined until it emerges as a prediction at the other end.

At each layer, two operations happen:
1. **Linear transformation:** The layer multiplies its input by a weight matrix and adds a bias — this rotates, scales, and shifts the data
2. **Non-linear activation:** A non-linear function is applied element-wise — this introduces the non-linearity that enables learning complex patterns

The forward pass at each layer can be visualized as:
- Input vector → Multiply by weights (linear combination of inputs) → Add bias → Apply activation → Output vector

In matrix terms, if you have a batch of $m$ input vectors stacked as rows of a matrix $\mathbf{X} \in \mathbb{R}^{m \times d_{in}}$, a layer computes:

$$\mathbf{Z} = \mathbf{X} \mathbf{W}^T + \mathbf{b}$$
$$\mathbf{A} = \sigma(\mathbf{Z})$$

The output $\mathbf{A} \in \mathbb{R}^{m \times d_{out}}$ becomes the input to the next layer.

## Why This Concept Matters

The forward pass is the fundamental computation at the heart of every neural network. Understanding it is crucial for:

- **Architecture Design:** Knowing how data shapes transform through layers helps design correct architectures
- **Debugging:** When predictions are wrong, checking intermediate activations (shapes, values, distributions) helps identify issues
- **Performance Optimization:** Forward pass time dominates inference; understanding it enables optimization
- **Custom Layers:** Implementing new layer types requires understanding the forward pass contract
- **Deployment:** Converting models to production formats (ONNX, TensorRT) requires tracing the forward pass

## Real World Examples

1. **Image Classification (ResNet):** A 224×224×3 image is forward-propagated through 50+ layers. Early conv layers transform spatial dimensions from 224→112→56→28→14→7 while increasing channels from 3→64→128→256→512. The final 7×7×512 tensor is pooled to a 2048-vector, which passes through a linear layer to produce class scores.

2. **Language Modeling (GPT):** A sequence of token IDs → embedding lookup (vocab_size × d_model) → positional encoding → 96 layers of self-attention + MLP → final layer norm → linear projection back to vocab_size → softmax. Each token's representation is a vector of size d_model (e.g., 12288 for GPT-3).

3. **Speech Recognition (DeepSpeech):** Raw audio spectrograms (frequency bins × time steps) → 1D convolutions along time → bidirectional LSTM → fully connected → character probabilities per time step.

## AI/ML Relevance

- **Inference:** The forward pass is the only computation needed for making predictions with a trained model
- **Computational Graph:** Frameworks like PyTorch and TensorFlow construct a computational graph during the forward pass for automatic differentiation
- **Activation Checkpointing:** Trading memory for compute by not storing intermediate activations during forward pass (recomputed during backward)
- **Quantization:** During forward pass, operations can be done in lower precision (INT8, FP16) for faster inference
- **JIT Compilation:** TorchScript and XLA compile the forward pass into optimized kernels

## Mathematical Explanation

### Single Neuron Forward Pass

$$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
$$a = \sigma(z)$$

### Layer Forward Pass (Vectorized)

$$\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}$$
$$\mathbf{a} = \sigma(\mathbf{z})$$

### Batch Forward Pass (Matrix)

$$\mathbf{Z} = \mathbf{X} \mathbf{W}^T + \mathbf{b}$$
$$\mathbf{A} = \sigma(\mathbf{Z})$$

where $\mathbf{X} \in \mathbb{R}^{m \times d_{in}}$, $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$, $\mathbf{b} \in \mathbb{R}^{d_{out}}$, and broadcasting adds $\mathbf{b}$ to each row of $\mathbf{X} \mathbf{W}^T$.

### Common Activation Forms

- **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$, output range $(0, 1)$
- **Tanh:** $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, output range $(-1, 1)$
- **ReLU:** $\sigma(z) = \max(0, z)$, output range $[0, \infty)$
- **Softmax:** $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$, output is a probability distribution

### Data Shape Transformation

For a batch of $m$ samples, input dimension $d_{in}$, output dimension $d_{out}$:

$$\underbrace{m \times d_{in}}_{\text{input}} \xrightarrow{\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}} \underbrace{m \times d_{out}}_{\text{output}}$$

## Code Examples

### Example 1: Forward Pass Step by Step

```python
import torch
import torch.nn as nn

# Single neuron forward pass (manual)
x = torch.tensor([0.5, -0.3, 0.8, 0.1])  # input: 4 features
w = torch.tensor([0.2, -0.5, 0.3, 0.7])  # weights
b = torch.tensor([0.1])                   # bias

z = torch.dot(w, x) + b
a = torch.sigmoid(z)
print(f"Single neuron: z={z.item():.4f}, a={a.item():.4f}")
# Output: Single neuron: z=0.6100, a=0.6480

# Layer forward pass (manual)
W = torch.randn(3, 4)  # 3 output neurons, 4 input features
b = torch.randn(3)     # 3 biases
z = W @ x + b
a = torch.relu(z)
print(f"Layer output: {a}")
# Output: Layer output: tensor([0.0000, 0.3514, 0.0000])

# Batch forward pass
X = torch.randn(10, 4)  # batch of 10 samples
Z = X @ W.T + b         # W.T because W is [out_features, in_features]
A = torch.relu(Z)
print(f"Batch output shape: {A.shape}")
# Output: Batch output shape: torch.Size([10, 3])
```

### Example 2: Forward Pass Through a Multi-Layer Network

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = torch.relu(self.fc1(x))
        print(f"After fc1: {x.shape}")
        x = torch.relu(self.fc2(x))
        print(f"After fc2: {x.shape}")
        x = self.fc3(x)
        print(f"Output (logits): {x.shape}")
        return x

model = SimpleMLP()
dummy_input = torch.randn(32, 784)
output = model(dummy_input)
# Output: Input shape: torch.Size([32, 784])
# Output: After fc1: torch.Size([32, 256])
# Output: After fc2: torch.Size([32, 128])
# Output: Output (logits): torch.Size([32, 10])
```

### Example 3: Forward Pass with Different Output Activations

```python
import torch
import torch.nn.functional as F

# Setup
dummy_input = torch.randn(4, 784)
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 10)
)

logits = model(dummy_input)

# Different output activations for different tasks
# Binary classification: sigmoid
binary_logits = logits[:, 0:1]
binary_probs = torch.sigmoid(binary_logits)
print(f"Binary classification probs: {binary_probs.shape}")
# Output: Binary classification probs: torch.Size([4, 1])

# Multi-class classification: softmax
multi_probs = F.softmax(logits, dim=1)
print(f"Multi-class probs (sum to 1 per row): {multi_probs.shape}")
# Output: Multi-class probs (sum to 1 per row): torch.Size([4, 10])
print(f"Row sums: {multi_probs.sum(dim=1)}")
# Output: Row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000])

# Multi-label classification: sigmoid per class
multi_label_probs = torch.sigmoid(logits)
print(f"Multi-label probs: {multi_label_probs.shape}")
# Output: Multi-label probs: torch.Size([4, 10])

# Regression: no activation (identity)
regression_output = logits[:, 0:1]
print(f"Regression output: {regression_output.shape}")
# Output: Regression output: torch.Size([4, 1])
```

## Common Mistakes

1. **Shape mismatch in matrix multiplication:** For $\mathbf{X} \mathbf{W}^T$, $\mathbf{X}$ is $(m, d_{in})$ and $\mathbf{W}$ is $(d_{out}, d_{in})$, so $\mathbf{W}^T$ is $(d_{in}, d_{out})$. The result is $(m, d_{out})$.

2. **Applying softmax to logits before CrossEntropyLoss:** PyTorch's `CrossEntropyLoss` applies softmax internally. If you apply softmax in the forward pass and then use CrossEntropyLoss, you get a double softmax.

3. **Using the wrong activation for the output layer:** Sigmoid for multi-class (should be softmax), linear for binary classification (should be sigmoid), ReLU for regression with negative outputs.

4. **Forgetting to flatten or reshape input:** CNNs expect (B, C, H, W), MLPs expect (B, D). Feeding raw images into a linear layer without flattening causes shape errors.

5. **Not accounting for batch dimension:** A single input sample should still have a batch dimension (1, D), not just (D,).

## Interview Questions

### Beginner

1. What is the forward pass in a neural network?
2. Write the equation for a single layer forward pass.
3. What happens to the shape of data as it passes through a linear layer from dimension $d_{in}$ to $d_{out}$ with batch size $m$?
4. What is the difference between logits and probabilities?
5. Name three output activations and when you would use each.

### Intermediate

1. Explain the matrix multiplication view of the forward pass. Why is it more efficient than processing one sample at a time?
2. How does the forward pass differ between training and inference for layers like dropout and batch normalization?
3. What is the computational complexity (FLOPs) of a forward pass through a linear layer with input dimension $d_{in}$ and output dimension $d_{out}$ for batch size $m$?
4. How would you implement a forward pass for a residual connection (skip connection)?
5. What is the purpose of activation checkpointing (gradient checkpointing) during the forward pass?

### Advanced

1. Derive the forward pass equations for a Transformer's multi-head attention layer, including the shape transformations.
2. Analyze the numerical stability of different activation functions during the forward pass — which combinations can cause overflow or underflow?
3. Design a custom forward pass that implements a mixture of experts (MoE) layer with top-k routing.

## Practice Problems

### Easy

1. Implement the forward pass of a single neuron manually using NumPy.
2. Trace the shape transformations through a network: Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 10) with batch size 64.
3. Compute the output of a softmax layer given logits [2.0, 1.0, 0.1].
4. Write a function that takes a batch of inputs and returns the forward pass through a 2-layer network using only PyTorch tensors (no nn.Module).
5. Verify that CrossEntropyLoss applied to logits produces the same result as NLLLoss applied to log-softmax.

### Medium

1. Implement a custom Linear layer from scratch (using torch.Tensor + autograd) and verify its forward pass matches nn.Linear.
2. Profile the forward pass time for different batch sizes and layer widths. Identify the fastest configuration for your hardware.
3. Implement a forward pass that uses mixed precision (FP16) for some layers and FP32 for others.
4. Add activation checkpointing to a deep network and measure the memory savings vs computational overhead.
5. Implement the forward pass of a residual block and show that the gradient flows better through the skip connection.

### Hard

1. Implement a forward pass for a Transformer encoder layer (self-attention + MLP + layer norm + residual connections).
2. Optimize a forward pass for inference by fusing operations (e.g., fold batch norm into preceding conv layer).
3. Implement a differentiable forward pass through a custom operation with a custom autograd Function.

## Solutions

### Easy 1
```python
import numpy as np
x = np.array([0.5, -0.3, 0.8])
w = np.array([0.2, 0.5, -0.1])
b = 0.1
z = np.dot(w, x) + b
a = 1 / (1 + np.exp(-z))
print(f"z={z:.4f}, a={a:.4f}")
# Output: z=0.1200, a=0.5300
```

### Easy 3
```python
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(probs)
# Output: tensor([0.6590, 0.2424, 0.0986])
```

### Medium 1
```python
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.W.T + self.b
```

## Related Concepts

- Backward Pass (DL-013)
- Activation Functions
- Computational Graphs
- Automatic Differentiation
- Matrix Operations

## Next Concepts

- Backpropagation
- Computational Graph Optimization
- Mixed Precision Training
- Activation Checkpointing
- Kernel Fusion

## Summary

The forward pass computes a neural network's output by sequentially applying linear transformations (weight matrix multiplication + bias addition) and non-linear activation functions. In matrix form, a batch of inputs is transformed from $m \times d_{in}$ to $m \times d_{out}$ through each layer. The forward pass is the only computation needed for inference and constructs the computational graph used for automatic differentiation during training. Understanding data shape transformations and activation choices is essential for correct architecture design.

## Key Takeaways

- Forward pass: input → [linear → activation] × layers → output
- Matrix view: $\mathbf{A} = \sigma(\mathbf{X} \mathbf{W}^T + \mathbf{b})$
- Shapes: $(m, d_{in}) \to (m, d_{out})$ per layer
- Output activation depends on task: softmax (classification), sigmoid (binary), identity (regression)
- Forward pass constructs the computational graph for automatic differentiation
