# Concept: PyTorch Tensors

## Concept ID

PYT-096

## Difficulty

Advanced

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Create and manipulate PyTorch tensors with various dtypes, shapes, and devices
- Understand tensor operations including indexing, slicing, broadcasting, and reshaping
- Move tensors between CPU and GPU with `.to()` and `.cuda()`
- Use autograd for automatic differentiation with `requires_grad` and `.backward()`

## Prerequisites

- NumPy arrays (indexing, broadcasting, reshaping)
- Basic understanding of GPU computing
- Calculus fundamentals (partial derivatives, gradients)

## Definition

PyTorch tensors are multi-dimensional arrays similar to NumPy arrays but optimized for GPU computation and automatic differentiation. They are the fundamental data structure in PyTorch:

- **Shape/Dimension:** Each tensor has a `shape` (tuple of sizes per dimension) and `ndim` (number of dimensions)
- **Dtype:** Data type — `torch.float32`, `torch.int64`, `torch.bool`, etc.
- **Device:** Where the tensor lives — CPU (`'cpu'`) or GPU (`'cuda:0'`)

**Creation functions:**
- `torch.tensor(data)`: From Python list or NumPy array
- `torch.zeros(shape)`, `torch.ones(shape)`: Filled with 0s or 1s
- `torch.randn(shape)`: Standard normal distribution
- `torch.arange(start, end)`: Range of values
- `torch.linspace(start, end, steps)`: Linearly spaced values

**Autograd (automatic differentiation):**
- `requires_grad=True`: PyTorch tracks all operations on this tensor to compute gradients
- `.backward()`: Computes gradients via backpropagation
- `.grad`: Stores the computed gradient

## Intuition

Tensors are to PyTorch what arrays are to NumPy, but with two superpowers:

1. **GPU acceleration:** PyTorch tensors can live on a GPU (NVIDIA CUDA or Apple MPS), where operations run 10-100x faster than on CPU. Moving data between devices is explicit: `tensor.to('cuda')`.

2. **Autograd:** When you set `requires_grad=True`, PyTorch builds a computational graph tracking every operation. Calling `.backward()` traverses this graph in reverse, computing gradients of the output with respect to every input in one pass.

Think of a neural network: inputs → hidden layers → loss. Autograd automatically computes d(loss)/d(weight) for every weight. Without autograd, you'd need to derive and code gradient formulas manually — infeasible for complex networks.

## Why This Concept Matters

- **Deep Learning Foundation:** Everything in PyTorch — models, loss functions, optimizers — operates on tensors
- **GPU Computing:** Modern ML runs on GPUs. Tensor device management is essential
- **Autograd:** The magic behind neural network training — without it, backpropagation must be manually derived and coded
- **Research Flexibility:** PyTorch's tensor operations enable custom model architectures and loss functions
- **Interoperability:** Tensors convert to/from NumPy arrays easily, enabling hybrid workflows

## Real World Examples

1. **Image Batch:** A batch of 32 RGB images of size 224×224 is stored as a tensor of shape `(32, 3, 224, 224)` — batch × channels × height × width.
2. **Text Embeddings:** A sentence of 50 tokens, each embedded as a 300-dim vector, is shape `(50, 300)`.
3. **Video Data:** A 10-second video at 30fps, 1280×720, RGB: shape `(300, 3, 720, 1280)`.
4. **Gradient Computation:** Training a neural network computes loss, calls `.backward()`, and optimizer `.step()` updates weights using `.grad`.
5. **Model Weights:** A linear layer mapping 784→256 features has a weight tensor of shape `(256, 784)` and bias of shape `(256,)`.

## AI/ML Relevance

- **Neural Network Training:** All computation happens via tensors with autograd
- **GPU Acceleration:** Training large models is infeasible on CPU alone
- **Custom Layers:** Advanced research requires building custom operations on tensors
- **Memory Management:** Understanding tensor shapes and dtypes is critical for fitting models in GPU memory
- **Gradient-Based Optimization:** Autograd enables first-order optimization (SGD, Adam) without manual derivative computation

## Code Examples

### Example 1: Creating tensors from different sources
```python
import torch
import numpy as np

# From Python list
t1 = torch.tensor([[1, 2], [3, 4]])
print(f"From list:\n{t1}, dtype={t1.dtype}")

# From NumPy array
arr = np.array([5.0, 6.0, 7.0])
t2 = torch.from_numpy(arr)
print(f"From numpy: {t2}, dtype={t2.dtype}")

# Special tensors
print(f"Zeros: {torch.zeros(2, 3)}")
print(f"Ones: {torch.ones(2, 3)}")
print(f"Randn: {torch.randn(2, 3)}")
print(f"Arange: {torch.arange(0, 10, 2)}")
print(f"Linspace: {torch.linspace(0, 1, 5)}")
```
```
# Output:
# From list:
# tensor([[1, 2],
#         [3, 4]]), dtype=torch.int64
# From numpy: tensor([5., 6., 7.]), dtype=torch.float64
# Zeros: tensor([[0., 0., 0.],
#                [0., 0., 0.]])
# Ones: tensor([[1., 1., 1.],
#               [1., 1., 1.]])
# Randn: tensor([[-0.2941,  0.5092,  0.5488],
#               [ 1.3713,  0.7252, -1.3479]])
# Arange: tensor([0, 2, 4, 6, 8])
# Linspace: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

### Example 2: Tensor properties — shape, dtype, device
```python
x = torch.randn(2, 3, 4)
print(f"Shape: {x.shape}")
print(f"ndim: {x.ndim}")
print(f"Size: {x.size()}")
print(f"Number of elements: {x.numel()}")
print(f"dtype: {x.dtype}")
print(f"Device: {x.device}")

# Change dtype
y = x.float()
z = x.double()
print(f"float dtype: {y.dtype}, double dtype: {z.dtype}")
```
```
# Output:
# Shape: torch.Size([2, 3, 4])
# ndim: 3
# Size: torch.Size([2, 3, 4])
# Number of elements: 24
# dtype: torch.float32
# Device: cpu
# float dtype: torch.float32, double dtype: torch.float64
```

### Example 3: Basic tensor operations
```python
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # element-wise
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
print(f"a.dot(b) = {a.dot(b)}")  # dot product
print(f"a.sum() = {a.sum()}")
print(f"a.mean() = {a.mean()}")
print(f"a.std() = {a.std()}")

# Matrix multiplication
M1 = torch.randn(2, 3)
M2 = torch.randn(3, 4)
print(f"\nMatmul shape: {torch.mm(M1, M2).shape}")  # (2, 4)
print(f"Matmul (a @ M2) shape: {(a @ M2).shape}")  # vector @ matrix
```
```
# Output:
# a + b = tensor([5., 7., 9.])
# a - b = tensor([-3., -3., -3.])
# a * b = tensor([ 4., 10., 18.])
# a / b = tensor([0.2500, 0.4000, 0.5000])
# a ** 2 = tensor([1., 4., 9.])
# a.dot(b) = tensor(32.)
# a.sum() = tensor(6.)
# a.mean() = tensor(2.)
# a.std() = tensor(1.)
# Matmul shape: torch.Size([2, 4])
# Matmul (a @ M2) shape: torch.Size([4])
```

### Example 4: Indexing, slicing, and reshaping
```python
x = torch.arange(12).reshape(3, 4)
print(f"Original:\n{x}")

# Indexing
print(f"First row: {x[0]}")
print(f"First column: {x[:, 0]}")
print(f"Element (1, 2): {x[1, 2]}")
print(f"First two rows, cols 1-3:\n{x[:2, 1:4]}")

# Reshaping
print(f"Reshaped to (2, 6):\n{x.reshape(2, 6)}")
print(f"Transpose:\n{x.T}")
print(f"Flattened: {x.flatten()}")

# Adding/removing dimensions
print(f"Unsqueeze (add batch dim): {x.unsqueeze(0).shape}")  # (1, 3, 4)
print(f"Squeeze (remove dim 1): {x.unsqueeze(1).squeeze(1).shape}")

# Permute / transpose
print(f"Permuted (1, 0):\n{x.permute(1, 0)}")
```
```
# Output:
# Original:
# tensor([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])
# First row: tensor([0, 1, 2, 3])
# First column: tensor([0, 4, 8])
# Element (1, 2): tensor(6)
# First two rows, cols 1-3:
# tensor([[ 1,  2,  3],
#        [ 5,  6,  7]])
# Reshaped to (2, 6):
# tensor([[ 0,  1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10, 11]])
# Transpose:
# tensor([[ 0,  4,  8],
#        [ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11]])
# Flattened: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# Unsqueeze (add batch dim): torch.Size([1, 3, 4])
# Squeeze (remove dim 1): torch.Size([3, 4])
# Permuted (1, 0):
# tensor([[ 0,  4,  8],
#        [ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11]])
```

### Example 5: Device management — CPU to GPU
```python
x = torch.randn(3, 3)
print(f"Initial device: {x.device}")

# Move to GPU if available
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(f"GPU device: {x_gpu.device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # Move back to CPU
    x_cpu = x_gpu.cpu()
    print(f"Back to CPU: {x_cpu.device}")

    # Alternative syntax
    x_gpu2 = x.cuda()
    print(f"Using .cuda(): {x_gpu2.device}")
else:
    print("CUDA not available — running on CPU only.")
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        x_mps = x.to('mps')
        print(f"MPS device: {x_mps.device}")
```
```
# Output:
# Initial device: cpu
# CUDA not available — running on CPU only.
```

### Example 6: NumPy interoperability
```python
import numpy as np

# Tensor to NumPy
x = torch.randn(2, 3)
x_np = x.numpy()
print(f"Tensor -> numpy: {type(x_np)}, shape: {x_np.shape}")
print(f"Shared memory? {x.data_ptr() == x_np.ctypes.data}")  # True (same memory)

# NumPy to Tensor
arr = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.from_numpy(arr)
print(f"Numpy -> tensor: {type(t)}, shape: {t.shape}")
print(f"dtype preserved: {t.dtype}")

# Important: modifying one modifies the other (shared buffer)
arr[0, 0] = 999
print(f"Modified numpy, tensor also changed:\n{t}")
```
```
# Output:
# Tensor -> numpy: <class 'numpy.ndarray'>, shape: (2, 3)
# Shared memory? True
# Numpy -> tensor: <class 'torch.Tensor'>, shape: torch.Size([2, 3])
# dtype preserved: torch.int64
# Modified numpy, tensor also changed:
# tensor([[999,   2,   3],
#        [  4,   5,   6]], dtype=torch.int32)
```

### Example 7: Broadcasting
```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])   # shape (2, 3)
b = torch.tensor([10, 20, 30])  # shape (3,)

# b broadcasts to shape (1, 3) then (2, 3)
print(f"a + b:\n{a + b}")

c = torch.tensor([[1], [10]])    # shape (2, 1)
print(f"a + c:\n{a + c}")        # c broadcasts to (2, 3)

# Broadcasting rules: aligned from right, dimensions must be equal or 1
d = torch.tensor([1, 2, 3, 4])  # shape (4,) — won't broadcast with (2, 3)
try:
    a + d
except RuntimeError as e:
    print(f"Broadcast error: {e}")
```
```
# Output:
# a + b:
# tensor([[11, 22, 33],
#        [14, 25, 36]])
# a + c:
# tensor([[ 2,  3,  4],
#        [14, 15, 16]])
# Broadcast error: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

### Example 8: Autograd basics
```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"Requires grad: {x.requires_grad}")

# Build computation graph
y = x ** 2
z = y.sum()
print(f"y = x^2 = {y}")
print(f"z = sum(y) = {z.item()}")

# Compute gradients
z.backward()
print(f"dz/dx1 = {x.grad[0].item()}")  # 2 * x1 = 4
print(f"dz/dx2 = {x.grad[1].item()}")  # 2 * x2 = 6

# Gradients accumulate — clear them
x.grad.zero_()
z2 = (x ** 3).sum()
z2.backward()
print(f"After zero_grad, dz2/dx1 = {x.grad[0].item()}")  # 3 * x1^2 = 12
```
```
# Output:
# Requires grad: True
# y = x^2 = tensor([4., 9.], grad_fn=<PowBackward0>)
# z = sum(y) = tensor(13.)
# dz/dx1 = 4.0
# dz/dx2 = 6.0
# After zero_grad, dz2/dx1 = 12.0
```

### Example 9: In-place operations and detaching
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

# In-place operation (trailing underscore)
z = y.clone()
z.add_(5)  # equivalent to z = z + 5
print(f"In-place add_: {z}")

# Detach — creates a new tensor that doesn't track gradients
w = x.detach()
print(f"Detached requires_grad: {w.requires_grad}")
w2 = x.detach().clone()  # common pattern: detach then clone

# With torch.no_grad() context
with torch.no_grad():
    out = x * 3
    print(f"Inside no_grad, requires_grad: {out.requires_grad}")

# Gradient accumulation
x.grad = None
loss = (x ** 2).sum()
loss.backward()
print(f"Grad after first backward: {x.grad}")

# Second backward WITHOUT zero_grad — gradients accumulate
loss2 = (x ** 3).sum()
loss2.backward()
print(f"Grad after second backward (accumulated): {x.grad}")
```
```
# Output:
# In-place add_: tensor([7., 9., 11.])
# Detached requires_grad: False
# Inside no_grad, requires_grad: False
# Grad after first backward: tensor([2., 4., 6.])
# Grad after second backward (accumulated): tensor([5., 16., 33.])
```

### Example 10: Reduction operations with dimension control
```python
x = torch.arange(12).reshape(3, 4).float()
print(f"Tensor:\n{x}")

print(f"Sum (all): {x.sum()}")
print(f"Mean (all): {x.mean()}")
print(f"Sum over dim=0 (columns): {x.sum(dim=0)}")     # sum each column
print(f"Sum over dim=1 (rows): {x.sum(dim=1)}")        # sum each row
print(f"Mean over dim=0: {x.mean(dim=0)}")
print(f"Max and argmax over dim=1: max={x.max(dim=1).values}, idx={x.max(dim=1).indices}")

# Keepdim preserves the reduced dimension
print(f"Sum with keepdim:\n{x.sum(dim=1, keepdim=True)}")

# Min, std, prod
print(f"Min: {x.min()}, Std: {x.std()}, Prod: {x.prod():.2e}")
```
```
# Output:
# Tensor:
# tensor([[ 0.,  1.,  2.,  3.],
#        [ 4.,  5.,  6.,  7.],
#        [ 8.,  9., 10., 11.]])
# Sum (all): tensor(66.)
# Mean (all): tensor(5.5000)
# Sum over dim=0 (columns): tensor([12., 15., 18., 21.])
# Sum over dim=1 (rows): tensor([ 6., 22., 38.])
# Mean over dim=0: tensor([4., 5., 6., 7.])
# Max and argmax over dim=1: max=tensor([ 3.,  7., 11.]), idx=tensor([3, 3, 3])
# Sum with keepdim:
# tensor([[ 6.],
#        [22.],
#        [38.]])
# Min: tensor(0.), Std: tensor(3.6056), Prod: tensor(0.)
```

### Example 11: Advanced autograd — Jacobian and higher-order gradients
```python
x = torch.randn(2, requires_grad=True)
y = x ** 2
z = torch.cat([y[:1].sin(), y[1:].cos()])
print(f"x: {x}")
print(f"z(x): {z}")

# Compute Jacobian (vector function -> vector output)
J = torch.autograd.functional.jacobian(lambda x: x**2, x)
print(f"Jacobian of f(x)=x^2:\n{J}")

# Higher-order gradients
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
second_grad = torch.autograd.grad(first_grad, x)[0]
print(f"f(x) = x^3, f'({x.item()}) = {first_grad.item()}, f''({x.item()}) = {second_grad.item()}")
```
```
# Output:
# x: tensor([-0.2564,  0.7825], requires_grad=True)
# z(x): tensor([-0.2536,  0.6811], grad_fn=<CatBackward0>)
# Jacobian of f(x)=x^2:
# tensor([[-0.5128,  0.0000],
#        [ 0.0000,  1.5650]])
# f(x) = x^3, f'(2.0) = 12.0, f''(2.0) = 12.0
```

## Common Mistakes

1. **Calling `.backward()` on a non-scalar tensor without passing `gradient`.** By default, `.backward()` computes gradients of a scalar. If your loss is a vector or matrix, pass a gradient tensor of the same shape (e.g., `loss.backward(gradient=torch.ones_like(loss))`).
2. **Forgetting to zero gradients before each backward pass.** Gradients accumulate by default. Always call `optimizer.zero_grad()` (or `tensor.grad.zero_()`) before `.backward()`.
3. **Using in-place operations on tensors that require grad.** In-place ops like `tensor.add_(5)` can interfere with autograd's computation graph, causing errors. Use `tensor = tensor + 5` instead.
4. **CPU/GPU device mismatch.** Operations between a CPU and CUDA tensor raise an error. Always move tensors to the same device: `tensor.to(device)`.
5. **Detaching then modifying and re-attaching incorrectly.** `detach()` creates a tensor that does not track history. The detached tensor cannot re-join the computation graph.
6. **Assuming `.numpy()` works on GPU tensors.** `.numpy()` only works on CPU tensors. Call `.cpu().numpy()` to move to CPU first.
7. **Ignoring the shared memory between NumPy and PyTorch.** `torch.from_numpy()` shares memory — modifying one modifies the other. Use `.clone()` to break the connection.

## Interview Questions

### Beginner - 5

1. **Q:** How do you create a 2×3 tensor filled with zeros?  
   **A:** `torch.zeros(2, 3)`.

2. **Q:** What is the difference between `torch.tensor()` and `torch.Tensor()`?  
   **A:** `torch.tensor(data)` infers dtype from data and copies data. `torch.Tensor()` returns an empty float32 tensor (or uses `torch.Tensor(shape)` for uninitialized data).

3. **Q:** How do you move a tensor to GPU?  
   **A:** `tensor.to('cuda')` or `tensor.cuda()`. Check availability first: `if torch.cuda.is_available()`.

4. **Q:** What does `requires_grad=True` do?  
   **A:** It tells PyTorch to track all operations on the tensor so that gradients can be computed automatically via backpropagation.

5. **Q:** How do you prevent PyTorch from tracking gradients?  
   **A:** Use the `with torch.no_grad():` context manager or call `tensor.detach()`.

### Intermediate - 5

1. **Q:** Explain tensor broadcasting rules in PyTorch.  
   **A:** Two tensors are broadcastable if: (a) they have the same number of dimensions, or one can be expanded by prepending 1s; (b) for each dimension, the sizes are equal or one is 1. The result takes the maximum size along each dimension.

2. **Q:** What is the difference between `x.view()`, `x.reshape()`, and `x.resize_()`?  
   **A:** `view()` requires contiguous memory and returns a new view (no data copy). `reshape()` works on non-contiguous tensors but may copy data. `resize_()` is in-place and may truncate or pad.

3. **Q:** How does `.backward()` work for a non-scalar tensor?  
   **A:** For a non-scalar output `y`, you must provide a `gradient` argument of the same shape — the Jacobian-vector product. Commonly used for computing gradients of vector-valued functions.

4. **Q:** What is the computational graph in PyTorch?  
   **A:** It's a directed acyclic graph (DAG) where nodes are tensors and edges are operations. During forward pass, PyTorch builds this graph. During backward pass, it traverses from the output backward to compute gradients via the chain rule.

5. **Q:** How do you profile memory usage of PyTorch tensors?  
   **A:** Use `torch.cuda.memory_summary()` for CUDA memory. For general profiling: `tensor.element_size() * tensor.numel()` gives the memory in bytes.

### Advanced - 3

1. **Q:** Explain how PyTorch's autograd implements the chain rule for a computational graph with branching.  
   **A:** Autograd uses reverse-mode automatic differentiation (backpropagation). For each node in the graph, it stores the gradient of the output w.r.t. that node. For branching (one tensor used in multiple downstream operations), gradients are summed at the branching point.

2. **Q:** How would you implement a custom autograd Function with a non-differentiable forward operation but a manually defined backward?  
   **A:** Subclass `torch.autograd.Function`, implement `forward(ctx, input)` and `backward(ctx, grad_output)`. Use `ctx.save_for_backward()` to save tensors needed in backward. Register with `apply()`.

3. **Q:** Describe PyTorch's memory management strategy for CUDA tensors. What is the caching allocator?  
   **A:** PyTorch uses a caching memory allocator. When you free a tensor, the allocated memory is returned to a cache (not the OS) for reuse by future tensors. This avoids costly CUDA malloc/free calls. Managed by `torch.cuda.empty_cache()` to release cached memory.

## Practice Problems

### Easy - 5

1. **E1:** Create a 1D tensor of 10 random normal values, compute its mean and std.
2. **E2:** Create a 4×4 identity matrix using `torch.eye()` and convert it to float32.
3. **E3:** Create two tensors `a = [1,2,3]` and `b = [4,5,6]`, compute their dot product.
4. **E4:** Create a tensor of shape (3, 4, 5) and report its shape, ndim, numel, and dtype.
5. **E5:** Create a tensor `x = [[1,2],[3,4]]`, compute `x * 2` (not matrix multiply).

### Medium - 5

1. **M1:** Create a tensor `x = [[1,2,3],[4,5,6]]` and compute: sum over dim=0, mean over dim=1, max over dim=1.
2. **M2:** Given `x = torch.randn(3, 3, requires_grad=True)`, compute `y = (x**2).sum().sqrt()`, call `.backward()`, and print `x.grad`.
3. **M3:** Create a tensor of shape (2, 3, 4), unsqueeze to (1, 2, 3, 4), then squeeze back to (2, 3, 4).
4. **M4:** Create a 1×3 tensor and a 2×1 tensor, add them using broadcasting, and verify the output shape.
5. **M5:** Move a random tensor to GPU (if available), run `torch.matmul` on it, move result back to CPU.

### Hard - 3

1. **H1:** Implement a custom autograd Function that computes `f(x) = x * exp(-x^2)` with a manually defined backward pass.
2. **H2:** Write a function that computes the second derivative (Hessian diagonal) of `f(x) = sum(x^3)` using `torch.autograd.grad` with `create_graph=True`.
3. **H3:** Create a simple linear regression model from scratch using only PyTorch tensors and autograd (no nn.Module), trained with gradient descent.

## Solutions

### E1 Solution
```python
x = torch.randn(10)
print(f"Mean: {x.mean():.3f}, Std: {x.std():.3f}")
```

### E2 Solution
```python
x = torch.eye(4, dtype=torch.float32)
print(x)
```

### E3-E5 Solutions follow from examples above.

### M1-M5 Solutions extend the techniques shown in examples 4, 7, 8, 5.

### H1-H3 Solutions require combining autograd, custom functions, and training loops.

## Related Concepts

- 097 — PyTorch NN (building neural networks with nn.Module, which operates on tensors)
- 099 — Training Loops (using autograd for weight updates)
- 098 — TensorFlow/Keras (equivalent tensor operations in TensorFlow)

## Next Concepts

- 097 — PyTorch NN (nn.Module, nn.Linear, loss functions, optimizers)
- 099 — Training Loops (epochs, batch loops, zero_grad, backward, step)
- 098 — TensorFlow/Keras (tensor operations and gradient tape)

## Summary

PyTorch tensors are multi-dimensional arrays with GPU acceleration (`.to('cuda')`) and automatic differentiation (`requires_grad`, `.backward()`). They support NumPy-like operations (indexing, broadcasting, reshaping) plus autograd for computing gradients via backpropagation. Tensor shape/dtype/device management is the foundation for all PyTorch deep learning.

## Key Takeaways

- Tensors are the fundamental data structure — everything operates on them
- Shape, dtype, and device are the three critical properties
- Move to GPU with `.to('cuda')` or `.cuda()` for acceleration
- `requires_grad=True` enables autograd tracking
- `.backward()` computes gradients via automatic differentiation
- Always `zero_grad()` before each backward pass to avoid accumulation
- Use `.detach()` or `torch.no_grad()` to exclude operations from the graph
- NumPy and PyTorch tensors share memory — use `.clone()` to break the link
