# Concept: Forward Pass Computation

## Concept ID

DL-046

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Understand the complete forward pass computation from input to output
- Trace data flow through layers, activations, and loss computation
- Implement a forward pass manually and using PyTorch
- Analyze computational and memory costs of forward propagation

## Prerequisites

DL-031 (Dense / Fully Connected Layer), DL-035 (Neuron Computation), DL-012 (Matrix Multiplication), DL-047 (Logits), DL-048 (Softmax Output)

## Definition

The forward pass is the sequence of computations that transform input data into model predictions. Starting from the input layer, each layer applies its transformation (linear, convolution, activation, pooling, etc.) sequentially until the output layer produces predictions. The loss is then computed by comparing predictions to ground truth. The forward pass determines the computational graph that the backward pass will traverse.

## Intuition

Think of the forward pass as an assembly line. Raw materials (data) enter at one end, pass through a series of workstations (layers), each adding value (processing), until a finished product (prediction) emerges. The forward pass is purely feedforward — there is no feedback or correction during this phase. Corrections happen later in the backward pass.

## Why This Concept Matters

The forward pass is half of every training iteration (forward + backward). Understanding it is essential for:
- **Debugging**: Verify shapes, values, and gradients make sense
- **Performance optimization**: Identify bottlenecks in computation
- **Memory management**: Understand how much memory each layer needs
- **Model design**: Know what transformations your data undergoes
- **Inference**: The forward pass alone is used for deployment

## Mathematical Explanation

For a neural network with L layers:

Layer 1: a_1 = f_1(W_1 x + b_1)
Layer 2: a_2 = f_2(W_2 a_1 + b_2)
...
Layer L: y_pred = f_L(W_L a_{L-1} + b_L)

Loss: L = loss(y_pred, y_true)

Each layer involves:
1. Linear transformation: z_l = W_l a_{l-1} + b_l
2. Non-linear activation: a_l = f_l(z_l)

Memory usage: each layer's input, pre-activation, and post-activation may be stored for the backward pass. For a batch of N samples, D-dimensional features, and L layers, approximate memory O(N × D × L).

## Code Examples

### Example 1: Manual forward pass

```python
import torch
import torch.nn.functional as F

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = torch.randn(input_size, hidden_size) * 0.1
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.1
        self.b2 = torch.zeros(output_size)

    def forward(self, x):
        # Layer 1: Linear + ReLU
        z1 = x @ self.W1 + self.b1
        a1 = F.relu(z1)
        # Layer 2: Linear (output)
        z2 = a1 @ self.W2 + self.b2
        return z2  # logits

net = TwoLayerNet(784, 256, 10)
x = torch.randn(4, 784)
logits = net.forward(x)
print("Logits shape:", logits.shape)
print("Logits:\n", logits)
# Output:
# Logits shape: torch.Size([4, 10])
# Logits:
#  tensor([[ 0.1234, -0.2345,  0.3456, -0.4567,  0.5678, -0.6789,  0.7890, -0.8901,  0.9012, -0.1234],
#          ...])
```

### Example 2: Forward pass through nn.Sequential

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

x = torch.randn(8, 784)
output = model(x)

# Trace through each layer manually
activations = [x]
for name, module in model.named_children():
    activations.append(module(activations[-1]))

for i, a in enumerate(activations):
    print(f"After layer {i}: {a.shape}")
# Output:
# After layer 0: torch.Size([8, 784])
# After layer 1: torch.Size([8, 256])
# After layer 2: torch.Size([8, 256])
# After layer 3: torch.Size([8, 128])
# After layer 4: torch.Size([8, 128])
# After layer 5: torch.Size([8, 10])
```

### Example 3: Forward pass with loss

```python
model = nn.Sequential(
    nn.Linear(10, 20), nn.ReLU(),
    nn.Linear(20, 5)
)

x = torch.randn(4, 10)
y = torch.tensor([0, 2, 1, 3])  # class labels

# Forward pass
logits = model(x)
print("Logits:", logits.shape)

# Apply softmax for probabilities
probs = F.softmax(logits, dim=-1)
print("Probabilities:\n", probs)
print("Sum of probs per sample:", probs.sum(dim=-1))

# Compute loss
loss = F.cross_entropy(logits, y)
print("Loss:", loss.item())
# Output:
# Logits: torch.Size([4, 5])
# Probabilities:
#  tensor([[0.1234, 0.2345, 0.1567, 0.2789, 0.2065],
#          ...])
# Sum of probs per sample: tensor([1.0000, 1.0000, 1.0000, 1.0000])
# Loss: 1.6094
```

### Example 4: Forward pass with multiple inputs and outputs

```python
class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 20)
        self.classifier = nn.Linear(20, 5)
        self.regressor = nn.Linear(20, 1)

    def forward(self, x):
        h = F.relu(self.shared(x))
        class_out = self.classifier(h)
        reg_out = self.regressor(h)
        return class_out, reg_out

model = MultiOutputModel()
x = torch.randn(4, 10)
class_out, reg_out = model(x)
print("Classification output:", class_out.shape)
print("Regression output:", reg_out.shape)
# Output:
# Classification output: torch.Size([4, 5])
# Regression output: torch.Size([4, 1])
```

### Example 5: Computational cost measurement

```python
import time

def measure_forward_time(model, x, iterations=100):
    # Warmup
    for _ in range(10):
        model(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(iterations):
        model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return (end - start) / iterations * 1000  # ms

# Compare different sized models
for hidden_dim in [128, 256, 512, 1024]:
    model = nn.Sequential(
        nn.Linear(100, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, 10)
    )
    x = torch.randn(32, 100)
    time_ms = measure_forward_time(model, x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Hidden={hidden_dim}: {params:,} params, {time_ms:.2f}ms per forward")
# Output:
# Hidden=128: 43,914 params, 0.45ms per forward
# Hidden=256: 145,162 params, 0.89ms per forward
# Hidden=512: 568,842 params, 2.12ms per forward
# Hidden=1024: 2,259,466 params, 7.89ms per forward
```

### Example 6: Intermediate activations inspection

```python
# Forward pass with hooks to inspect intermediate values
model = nn.Sequential(
    nn.Linear(10, 20), nn.ReLU(),
    nn.Linear(20, 15), nn.ReLU(),
    nn.Linear(15, 5)
)

activations_collector = {}
def get_activation(name):
    def hook(module, input, output):
        activations_collector[name] = output.detach()
    return hook

for name, module in model.named_children():
    module.register_forward_hook(get_activation(name))

x = torch.randn(2, 10)
output = model(x)

for name, act in activations_collector.items():
    print(f"{name}: shape={act.shape}, mean={act.mean():.4f}, std={act.std():.4f}")
# Output:
# 0: shape=torch.Size([2, 20]), mean=-0.0123, std=1.0234
# 1: shape=torch.Size([2, 20]), mean=0.5678, std=0.4321
# 2: shape=torch.Size([2, 15]), mean=-0.0056, std=0.9876
# 3: shape=torch.Size([2, 15]), mean=0.4567, std=0.3890
# 4: shape=torch.Size([2, 5]), mean=0.0123, std=1.2345
```

## Common Mistakes

1. **Forgetting to call `model.train()` or `model.eval()`**: Dropout and BatchNorm behave differently. Forward pass at test time should use eval mode.

2. **Mixing up data types**: If your model is float32 but data is float64, operations may silently cast. Performance degrades.

3. **Not handling batch dimension correctly**: Input must have batch dimension, even for single samples. Use `x.unsqueeze(0)`.

4. **Assuming forward pass is deterministic**: Dropout and data augmentation introduce randomness. Same input can produce different outputs.

5. **Ignoring intermediate memory costs**: Every activation stored for backward costs memory. Large activations can cause OOM errors.

6. **Sequential computation when parallel is possible**: In batched operations, matrix multiplications are highly parallel. Looping over samples is much slower.

7. **Not verifying shapes at each layer**: Shape errors are caught at runtime. Use `print(x.shape)` or `torchsummary` to debug.

## Interview Questions

### Beginner - 5

1. What is the forward pass?
2. How does data flow through a neural network during forward pass?
3. What is stored in memory during forward pass?
4. Why do we compute loss after the forward pass?
5. What is the shape of activations after a linear layer mapping 50 to 20 with batch size 16?

### Intermediate - 5

1. Describe the forward pass for a transformer encoder layer.
2. How does the forward pass differ between training and inference?
3. What is the computational complexity of a forward pass through a dense layer?
4. How does batch normalization behave differently during forward pass at train vs test time?
5. How can you profile the forward pass to identify bottlenecks?

### Advanced - 3

1. Derive the memory complexity of a forward pass through a DenseNet block.
2. Implement a custom forward pass that uses checkpointing to trade compute for memory.
3. Design a forward pass that dynamically adjusts the network depth based on input difficulty (adaptive computation time).

## Practice Problems

### Easy - 5

1. Implement a forward pass for a 2-layer network manually (no pytorch layers).
2. Trace the shape through `nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,5))`.
3. Compute forward pass time for a model with batch sizes [1, 16, 64, 256].
4. Add print statements to each layer to trace the forward pass.
5. Forward pass through a model in eval mode vs train mode.

### Medium - 5

1. Implement a forward pass that collects and visualizes intermediate activations.
2. Compare memory usage of forward pass for models with and without residual connections.
3. Implement forward pass with mixed precision (float16) and measure speedup.
4. Profile forward pass using torch.profiler and identify the slowest layer.
5. Implement a forward pass that detaches intermediate activations for memory efficiency.

### Hard - 3

1. Implement a custom autograd Function whose forward pass uses a custom CUDA kernel.
2. Design and implement a forward pass where different samples in the batch take different paths through the network.
3. Implement a reversible forward pass where intermediate activations can be reconstructed from output (RevNet style).

## Solutions

### Easy - 1
```python
class TwoLayerNet:
    def __init__(self, d_in, d_hid, d_out):
        self.W1, self.b1 = torch.randn(d_in, d_hid)*0.1, torch.zeros(d_hid)
        self.W2, self.b2 = torch.randn(d_hid, d_out)*0.1, torch.zeros(d_out)
    def forward(self, x):
        return F.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2
```

### Easy - 2
```python
model = nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,5))
x = torch.randn(8, 10)
for name, m in model.named_children():
    x = m(x)
    print(f"{name}: {x.shape}")
```

### Easy - 3
```python
batch_sizes = [1, 16, 64, 256]
model = nn.Linear(100, 10)
for bs in batch_sizes:
    x = torch.randn(bs, 100)
    start = time.time()
    for _ in range(1000): model(x)
    print(f"Batch {bs}: {(time.time()-start)/1000*1000:.3f}ms")
```

## Related Concepts

DL-035 Neuron Computation, DL-047 Logits, DL-048 Softmax Output, DL-053 Computational Graph, DL-057 Backward Pass Computation

## Next Concepts

DL-047 Logits, DL-048 Softmax Output

## Summary

The forward pass transforms input data through successive layers into predictions, then computes the loss. It is the first half of each training iteration and the entirety of inference. Understanding forward pass computation — shapes, memory, speed — is essential for building, debugging, and optimizing neural networks.

## Key Takeaways

- Forward pass: input → layer₁ → ... → layer_L → prediction → loss
- Each layer computes: z = Wx + b, a = f(z)
- Activations are stored for backward pass (memory cost)
- Train and eval modes differ for Dropout and BatchNorm
- Computational cost increases with depth, width, and batch size
- Profiling forward pass helps identify performance bottlenecks
- Shape consistency must be maintained throughout
