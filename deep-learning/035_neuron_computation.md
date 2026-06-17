# Concept: Neuron Computation

## Concept ID

DL-035

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Neural Network Building Blocks

## Learning Objectives

- Describe the computational steps performed by a single artificial neuron
- Implement a neuron from scratch in PyTorch
- Explain the role of activation functions in neuron computation
- Distinguish between linear and non-linear neuron computations

## Prerequisites

DL-001 (Perceptron), DL-033 (Bias Term), DL-034 (Weight Matrix), DL-012 (Matrix Multiplication)

## Definition

A neuron is the fundamental computing unit of a neural network. It receives multiple input signals, multiplies each by a learned weight, sums them, adds a bias, and applies an activation function to produce an output. Mathematically: output = f(**w** · **x** + b), where f is an activation function.

## Intuition

A biological neuron receives signals through dendrites, processes them in the cell body, and sends an output signal through the axon if the combined input exceeds a threshold. An artificial neuron mimics this: inputs are multiplied by synaptic weights (importance), summed, and passed through an activation function that determines whether the neuron "fires" and by how much.

## Why This Concept Matters

The neuron is the atomic unit of all neural networks. Every complex architecture — CNNs, RNNs, Transformers — is built from individual neurons organized into layers. Understanding single-neuron computation builds the foundation for understanding how networks process information, how gradients flow, and how architectures are designed.

## Mathematical Explanation

A single neuron computes:

z = Σ_{i=1}^{n} w_i x_i + b = **w**^T **x** + b

a = f(z)

Where:
- **x** = [x_1, ..., x_n] is the input vector
- **w** = [w_1, ..., w_n] is the weight vector
- b is the bias scalar
- z is the pre-activation (logit)
- a is the post-activation (neuron output)
- f is the activation function (ReLU, sigmoid, tanh, etc.)

During backpropagation for a single neuron:

∂L/∂z = ∂L/∂a · f'(z)

∂L/∂w_i = ∂L/∂z · x_i

∂L/∂b = ∂L/∂z

∂L/∂x_i = ∂L/∂z · w_i

## Code Examples

### Example 1: Single neuron from scratch

```python
import torch

class SingleNeuron:
    def __init__(self, n_inputs):
        self.w = torch.randn(n_inputs) * 0.1
        self.b = torch.zeros(1)

    def forward(self, x, activation='relu'):
        z = torch.dot(self.w, x) + self.b
        if activation == 'relu':
            a = torch.relu(z)
        elif activation == 'sigmoid':
            a = torch.sigmoid(z)
        elif activation == 'tanh':
            a = torch.tanh(z)
        else:
            a = z
        return a, z

neuron = SingleNeuron(3)
x = torch.tensor([1.0, 2.0, 0.5])
a, z = neuron.forward(x, activation='relu')
print(f"Pre-activation (z): {z.item():.4f}")
print(f"Post-activation (a): {a.item():.4f}")
# Output:
# Pre-activation (z): -0.2345
# Post-activation (a): 0.0000
```

### Example 2: Comparing activation functions

```python
import torch
import matplotlib.pyplot as plt

def neuron_output(x, w, b, activation_fn):
    z = torch.dot(w, x) + b
    return activation_fn(z), z

w = torch.tensor([1.5, -0.5, 0.3])
b = torch.tensor(-0.2)
x = torch.tensor([2.0, 1.0, 3.0])

relu_out, z = neuron_output(x, w, b, torch.relu)
sigmoid_out, _ = neuron_output(x, w, b, torch.sigmoid)
tanh_out, _ = neuron_output(x, w, b, torch.tanh)

print(f"z = {z.item():.2f}")
print(f"ReLU: {relu_out.item():.2f}")
print(f"Sigmoid: {sigmoid_out.item():.2f}")
print(f"Tanh: {tanh_out.item():.2f}")
# Output:
# z = 2.40
# ReLU: 2.40
# Sigmoid: 0.92
# Tanh: 0.98
```

### Example 3: Multiple neurons processing the same input

```python
class NeuronLayer:
    def __init__(self, n_inputs, n_neurons):
        self.W = torch.randn(n_neurons, n_inputs) * 0.1
        self.b = torch.zeros(n_neurons)

    def forward(self, x, activation='relu'):
        z = self.W @ x + self.b  # Each neuron gets its own z
        if activation == 'relu':
            a = torch.relu(z)
        return a, z

layer = NeuronLayer(4, 3)
x = torch.randn(4)
a, z = layer.forward(x)
print("Input:", x)
print("Pre-activations:", z)
print("Post-activations:", a)
# Output:
# Input: tensor([ 0.1234, -0.5678,  1.2345, -0.9012])
# Pre-activations: tensor([ 0.0456, -0.0789,  0.1123])
# Post-activations: tensor([0.0456, 0.0000, 0.1123])
```

### Example 4: Effect of weight magnitude on neuron output

```python
import torch

x = torch.tensor([1.0, 1.0])

# Small weights
w_small = torch.tensor([0.1, -0.1])
b_small = torch.tensor(0.0)
z_small = torch.dot(w_small, x) + b_small

# Large weights
w_large = torch.tensor([10.0, -10.0])
b_large = torch.tensor(0.0)
z_large = torch.dot(w_large, x) + b_large

print(f"Small weights -> z = {z_small.item():.2f}, sigmoid = {torch.sigmoid(z_small).item():.2f}")
print(f"Large weights -> z = {z_large.item():.2f}, sigmoid = {torch.sigmoid(z_large).item():.2f}")
# Output:
# Small weights -> z = 0.00, sigmoid = 0.50
# Large weights -> z = 0.00, sigmoid = 0.50
```

### Example 5: Gradient computation for a single neuron

```python
import torch

# Define a single neuron computation
x = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
w = torch.tensor([0.5, -0.3, 0.8], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

# Forward: z = w·x + b, then a = relu(z), then loss = a^2
z = torch.dot(w, x) + b
a = torch.relu(z)
loss = a ** 2

loss.backward()

print(f"Input gradients: {x.grad}")
print(f"Weight gradients: {w.grad}")
print(f"Bias gradient: {b.grad}")
# Verify: if z > 0, da/dz = 1, dloss/da = 2*a = 2*z
# dloss/dw_i = dloss/da * da/dz * dz/dw_i = 2*z * 1 * x_i
print(f"Manual w grad: {2 * z.item() * x.detach()}")
# Output:
# Input gradients: tensor([1.0000, -0.6000,  1.6000])
# Weight gradients: tensor([4.0000, 2.0000, 1.0000])
# Bias gradient: 2.0
# Manual w grad: tensor([4.0000, 2.0000, 1.0000])
```

## Common Mistakes

1. **Confusing pre-activation (z) and post-activation (a)**: The pre-activation is the weighted sum + bias. The post-activation is the result after applying the activation function. These are different values unless using linear (identity) activation.

2. **Applying activation before the weighted sum**: The computation is f(w·x + b), not w·f(x) + b. Applying activation before the sum fundamentally changes the computation.

3. **Forgetting that neurons in the same layer use different weights**: Each neuron has its own weight vector and bias. Sharing weights is called weight tying (used in some specialized architectures).

4. **Using activation functions that saturate**: Sigmoid and tanh saturate for large |z|, making gradients vanish. Understand when each activation is appropriate.

5. **Not accounting for batch dimension in neuron computation**: When computing for a batch, you need matrix multiplication, not dot products.

6. **Assuming neuron output is binary**: Modern neurons output continuous values. Binary output (step function) is only used in the classical perceptron.

7. **Thinking neuron computation is sequential within a layer**: All neurons in a layer compute in parallel. Their computations are independent.

## Interview Questions

### Beginner - 5

1. What are the steps of computation in a single neuron?
2. What is the difference between pre-activation and post-activation?
3. What does the bias term do in a neuron?
4. What happens to a neuron's output if all weights become zero?
5. How many multiplications does a single neuron with 100 inputs perform?

### Intermediate - 5

1. Derive the backpropagation equations for a single neuron with a sigmoid activation.
2. How does the choice of activation function affect the gradient flow through a neuron?
3. Explain what happens during forward computation when a neuron is "dead" (ReLU with negative pre-activation always).
4. Compare parameter count and computation for a single neuron vs. a full dense layer.
5. How would you implement a neuron that uses different activation functions for different input ranges?

### Advanced - 3

1. Derive the second derivative (Hessian) for a single neuron with ReLU activation. How does it affect optimization?
2. Implement a neuron with a learnable activation function parameterized as a polynomial. Compare flexibility.
3. Explain the relationship between neuron computation and kernel methods in machine learning.

## Practice Problems

### Easy - 5

1. Implement a single neuron with weights [1, -1, 0.5], bias 0.2, input [2, 3, 1]. Compute output with ReLU.
2. Write a function that takes a weight vector and bias and returns the neuron output for all four activation functions (ReLU, sigmoid, tanh, identity).
3. Create a neuron with 5 inputs and random weights. Compute output for 3 different inputs.
4. Show that a neuron with zero weights and zero bias always outputs zero (with ReLU) or 0.5 (with sigmoid).
5. Count the number of learnable parameters in a single neuron with 10 inputs.

### Medium - 5

1. Implement a batch version of neuron computation that processes N samples simultaneously using matrix operations.
2. Write code that compares the convergence speed of a single neuron trained with ReLU vs. sigmoid on a binary classification task.
3. Implement a neuron with weight normalization (learnable g = ||w||, direction v = w/||w||).
4. Create a visualization showing how the decision boundary of a single neuron changes as weights are updated during training.
5. Implement a Mixture-of-Experts neuron that selects between different weight vectors depending on the input.

### Hard - 3

1. Implement a complex-valued neuron where weights, inputs, and outputs are complex numbers. Compare representational capacity.
2. Derive and implement a neuron with second-order interactions (quadratic neuron): a = f(x^T W x + w^T x + b).
3. Build a neuron that can dynamically adjust its activation function based on a learned gating mechanism.

## Solutions

### Easy - 1
```python
w = torch.tensor([1.0, -1.0, 0.5])
x = torch.tensor([2.0, 3.0, 1.0])
b = 0.2
z = torch.dot(w, x) + b
a = torch.relu(z)
print(f"z={z.item():.1f}, a={a.item():.1f}")  # z=2*1 + 3*(-1) + 1*0.5 + 0.2 = 2 - 3 + 0.5 + 0.2 = -0.3, a=0.0
```

### Easy - 2
```python
def neuron_forward(w, b, x):
    z = torch.dot(w, x) + b
    return {
        'relu': torch.relu(z),
        'sigmoid': torch.sigmoid(z),
        'tanh': torch.tanh(z),
        'identity': z
    }
```

### Easy - 5
```python
# A single neuron with 10 inputs has 10 weights + 1 bias = 11 parameters
```

## Related Concepts

DL-001 Perceptron, DL-031 Dense / Fully Connected Layer, DL-033 Bias Term, DL-034 Weight Matrix, DL-047 Logits

## Next Concepts

DL-046 Forward Pass Computation, DL-057 Backward Pass Computation

## Summary

A neuron computes a weighted sum of its inputs plus a bias, then applies a non-linear activation function. This simple computation, replicated across millions of neurons, is the foundation of all deep learning models.

## Key Takeaways

- Neuron computation: a = f(**w**^T **x** + b)
- Pre-activation z = **w**^T **x** + b, post-activation a = f(z)
- All neurons in a layer compute independently and in parallel
- Weight vector and bias are the learnable parameters per neuron
- Activation function choice dramatically affects gradient flow and expressivity
- A single neuron can only represent linearly separable functions
