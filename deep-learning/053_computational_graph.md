# Concept: Computational Graph

## Concept ID

DL-053

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Understand the structure and representation of computational graphs
- Trace operations through a computational graph
- Implement computational graph construction using PyTorch
- Analyze memory and computation trade-offs in graph design

## Prerequisites

DL-046 (Forward Pass Computation), DL-012 (Matrix Multiplication), DL-054 (Directed Acyclic Graph)

## Definition

A computational graph is a directed graph where nodes represent operations (add, multiply, nonlinearities) or variables (inputs, parameters, intermediate values) and edges represent data dependencies. The forward pass traverses the graph from inputs to outputs, computing results. The backward pass traverses in reverse, computing gradients via the chain rule. PyTorch builds a computational graph dynamically during the forward pass.

## Intuition

Think of a computational graph as a recipe. Each step (operation) takes ingredients (tensors) and produces a dish (new tensor). The recipe starts with raw ingredients (inputs, parameters) and ends with the final dish (loss). The backward pass is like figuring out which ingredient changes most affect the final taste. 

## Why This Concept Matters

The computational graph is the foundation of automatic differentiation:
- **PyTorch's dynamic graphs**: Built on-the-fly every forward pass
- **TensorFlow's static graphs**: Defined first, then executed
- **Gradient computation**: Backward pass traverses the graph
- **Optimization**: Graph transformations reduce memory and compute
- **Debugging**: Understanding the graph helps identify errors

## Mathematical Explanation

Nodes in the graph:
- Leaf nodes: inputs, parameters (require grad)
- Operation nodes: functions that transform tensors
- Intermediate nodes: results of operations
- Output node: final loss

Edges represent data flow from inputs to outputs.

For a node z = f(x, y):
- Forward: compute z from x, y
- Backward: compute ∂L/∂x and ∂L/∂y from ∂L/∂z and the local gradients ∂z/∂x, ∂z/∂y

The graph enables the chain rule: to compute ∂L/∂w for any parameter w, we traverse the path from L to w multiplying local Jacobians.

## Code Examples

### Example 1: Building a simple computational graph

```python
import torch

# Computational graph is built automatically
a = torch.tensor([2.0], requires_grad=True)  # Leaf node (parameter)
b = torch.tensor([3.0], requires_grad=True)  # Leaf node
c = torch.tensor([1.0])  # Leaf node (no grad needed)

d = a * b       # Multiplication node
e = d + c       # Addition node
loss = e ** 2   # Power node (output)

print("Computational graph nodes:")
print(f"  a: {a}, grad_fn={a.grad_fn}")      # None (leaf)
print(f"  d = a * b = {d}, grad_fn={d.grad_fn}")  # MulBackward0
print(f"  e = d + c = {e}, grad_fn={e.grad_fn}")  # AddBackward0
print(f"  loss = e^2 = {loss}, grad_fn={loss.grad_fn}")  # PowBackward0

loss.backward()
print(f"\nGradients:")
print(f"  ∂loss/∂a = {a.grad}")
print(f"  ∂loss/∂b = {b.grad}")
# Output:
# a: tensor([2.], requires_grad=True), grad_fn=None
# d = a * b = 6.0, grad_fn=<MulBackward0>
# e = d + c = 7.0, grad_fn=<AddBackward0>
# loss = e^2 = 49.0, grad_fn=<PowBackward0>
#
# Gradients:
#   ∂loss/∂a = tensor([42.])
#   ∂loss/∂b = tensor([28.])
```

### Example 2: Visualizing the graph

```python
# Print the graph structure recursively
def print_graph(node, indent=0):
    prefix = "  " * indent
    if hasattr(node, 'grad_fn') and node.grad_fn is not None:
        fn = node.grad_fn
        print(f"{prefix}{node.shape} <- {fn.__class__.__name__}")
        for parent in fn.next_functions:
            if parent[0] is not None:
                print_graph(parent[0].variable, indent + 1)
    else:
        print(f"{prefix}{node.shape} [leaf, requires_grad={node.requires_grad}]")

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = torch.relu(a @ b + torch.randn(3))
loss = c.sum()
print("Computational graph tree:")
print_graph(loss)
# Output:
# Computational graph tree:
# torch.Size([]) <- SumBackward0
#   torch.Size([3]) <- ReluBackward0
#     torch.Size([3]) <- AddBackward0
#       torch.Size([3]) <- MmBackward0
#         torch.Size([3]) [leaf, requires_grad=True]
#         torch.Size([3]) [leaf, requires_grad=True]
#       torch.Size([3]) [leaf, requires_grad=False]
```

### Example 3: Dynamic graph construction

```python
# PyTorch builds a new graph each forward pass
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# First graph
y1 = x ** 2
loss1 = y1.sum()
print(f"Graph 1 grad_fn: {loss1.grad_fn}")

# Second graph (different structure)
y2 = x ** 3
y3 = y2 + x
loss2 = y3.mean()
print(f"Graph 2 grad_fn: {loss2.grad_fn}")

# Third graph (conditional structure)
if x.sum().item() > 5:
    y4 = x ** 2
else:
    y4 = x * 2
loss3 = y4.sum()
print(f"Graph 3 grad_fn: {loss3.grad_fn}")
# Output:
# Graph 1 grad_fn: <SumBackward0>
# Graph 2 grad_fn: <MeanBackward0>
# Graph 3 grad_fn: <SumBackward0>
```

### Example 4: Graph memory tracking

```python
import torch

x = torch.randn(100, 784)
w1 = torch.randn(784, 256, requires_grad=True)
w2 = torch.randn(256, 128, requires_grad=True)
w3 = torch.randn(128, 10, requires_grad=True)

# Each forward + backward builds and destroys a graph
import gc
gc.collect()

# Track graph nodes
def count_graph_nodes(tensor):
    count = 0
    visited = set()
    def _count(t):
        nonlocal count
        if id(t) in visited:
            return
        visited.add(id(t))
        if hasattr(t, 'grad_fn') and t.grad_fn is not None:
            count += 1
            for parent, _ in t.grad_fn.next_functions:
                if parent is not None and hasattr(parent, 'variable'):
                    _count(parent.variable)
    _count(tensor)
    return count

h1 = torch.relu(x @ w1)
h2 = torch.relu(h1 @ w2)
logits = h2 @ w3
loss = F.cross_entropy(logits, torch.randint(0, 10, (100,)))

print(f"Graph has ~{count_graph_nodes(loss)} operation nodes")
print(f"Leaf parameters: 3 weight matrices")
# Output:
# Graph has ~12 operation nodes
# Leaf parameters: 3 weight matrices
```

### Example 5: Custom operation with graph

```python
class CustomSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x

# Use custom op in graph
x = torch.tensor([3.0, 4.0], requires_grad=True)
y = CustomSquare.apply(x)
loss = y.sum()
loss.backward()
print(f"Gradient: {x.grad}")  # [2*3, 2*4] = [6, 8]
print(f"Custom op in graph: {y.grad_fn}")
# Output:
# Gradient: tensor([6., 8.])
# Custom op in graph: <CustomSquareBackward0>
```

### Example 6: Detaching from graph

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Part of graph
y1 = x ** 2

# Detached from graph (no gradient tracking)
y2 = y1.detach()
print(f"y1 requires grad: {y1.requires_grad}")
print(f"y2 requires grad: {y2.requires_grad}")

# y2 can be used without affecting gradient
y3 = y2 + x
loss = y3.sum()
loss.backward()
print(f"x.grad = {x.grad}")  # Only gradient from y3 path, not y1
# Output:
# y1 requires grad: True
# y2 requires grad: False
# x.grad = tensor([1., 1.])
```

## Common Mistakes

1. **Modifying tensors in-place after gradient computation**: In-place operations (like `x += 1`) break the computational graph. Use `x = x + 1` instead.

2. **Forgetting to zero gradients**: Gradients accumulate in the graph. Always call `optimizer.zero_grad()` before each backward pass.

3. **Reusing tensors across graphs incorrectly**: If you reuse a tensor from a previous graph, it may retain references. Use `.detach()` or build a new graph.

4. **Building too complex graphs**: Very deep graphs consume memory proportional to depth. Use checkpointing to trade compute for memory.

5. **Not handling non-differentiable operations**: Some operations (sorting, indexing, thresholding) break gradients. Be aware of when gradients won't flow.

6. **Confusing leaf and non-leaf tensors**: Only leaf tensors (inputs, parameters) accumulate gradients. Non-leaf tensors have their gradients freed after backward.

7. **Forgetting that the graph is destroyed after backward**: After `loss.backward()`, the intermediate graph nodes are freed. Store needed values before backward.

## Interview Questions

### Beginner - 5

1. What is a computational graph?
2. How does PyTorch build a computational graph?
3. What are leaf nodes in a computational graph?
4. Why is the graph destroyed after backward?
5. What does `requires_grad=True` do?

### Intermediate - 5

1. Explain how the backward pass uses the computational graph.
2. What is the difference between dynamic and static computational graphs?
3. How does `detach()` affect the computational graph?
4. How are gradients accumulated when a variable is used multiple times?
5. What is the memory cost of storing a computational graph?

### Advanced - 3

1. Implement a custom autograd Function with a non-trivial backward pass.
2. Explain how gradient checkpointing modifies the computational graph.
3. Design a computational graph that supports second-order gradients (double backward).

## Practice Problems

### Easy - 5

1. Build a computational graph for `(a + b) * c` and compute gradients.
2. Print the `grad_fn` for each tensor in a simple expression.
3. Use `detach()` to remove a tensor from the graph.
4. Compute second-order gradients using `create_graph=True`.
5. Build a graph with a custom autograd Function.

### Medium - 5

1. Visualize the computation graph using `torchviz` or manual traversal.
2. Implement gradient checkpointing for a sequence of 10 linear layers.
3. Compare memory usage of a single graph vs. multiple sub-graphs.
4. Implement a model that uses the gradient of the gradient (Meta-learning).
5. Trace the graph traversal during backward and measure per-node time.

### Hard - 3

1. Implement a reversible computation graph where intermediate activations can be reconstructed.
2. Build a graph that supports higher-order optimization (e.g., computing the Hessian-vector product).
3. Design a graph that dynamically prunes operations based on gradient magnitude.

## Solutions

### Easy - 1
```python
a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([3.], requires_grad=True)
c = torch.tensor([4.], requires_grad=True)
y = (a + b) * c
y.backward()
print(a.grad, b.grad, c.grad)  # c=4, c=4, a+b=5
```

### Easy - 2
```python
x = torch.tensor([1.], requires_grad=True)
y = torch.sin(x ** 2)
print(f"y.grad_fn: {y.grad_fn}")
print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")
```

### Easy - 3
```python
x = torch.tensor([2.], requires_grad=True)
y1 = x ** 2
y2 = y1.detach()  # detached from graph
y3 = y2 + x
y3.backward()
print(x.grad)  # only 1 from y3 path, not 4 from y1 path
```

## Related Concepts

DL-054 Directed Acyclic Graph, DL-055 Forward Mode Autodiff, DL-063 Automatic Differentiation, DL-064 Reverse Mode Autodiff

## Next Concepts

DL-054 Directed Acyclic Graph, DL-063 Automatic Differentiation

## Summary

A computational graph represents a neural network's computation as a directed graph of operations on tensors. PyTorch builds this graph dynamically during forward execution and uses it to compute gradients during backward. Understanding computational graphs is essential for debugging, optimization, and advanced gradient techniques.

## Key Takeaways

- Nodes = operations, Edges = data dependencies
- PyTorch builds dynamic graphs (new graph each forward)
- Forward pass computes values; backward pass computes gradients
- Leaf nodes = inputs/parameters; non-leaf = intermediates
- Graph is destroyed after backward (unless retain_graph=True)
- detach() removes a tensor from the graph
- Memory scales with graph depth (number of intermediate nodes)
- Custom autograd Functions extend the graph
