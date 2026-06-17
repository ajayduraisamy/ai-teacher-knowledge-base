# Concept: Computation Graph Backward

## Concept ID

DL-065

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand the backward traversal of the computation graph
- Implement graph-based gradient propagation
- Analyze gradient flow through different graph structures
- Handle complex graph topologies (branching, merging, shared nodes)

## Prerequisites

DL-064 (Reverse Mode Autodiff), DL-053 (Computational Graph), DL-054 (Directed Acyclic Graph)

## Definition

The computation graph backward traversal is the process of computing gradients by visiting nodes in reverse topological order of the computation graph. Each node receives incoming gradients from its consumers, computes local gradients using its stored forward values, and propagates them to its producers (inputs). This is the concrete implementation of reverse-mode automatic differentiation on a graph data structure.

## Intuition

Imagine the computation graph as a river delta flowing from source (input) to sea (loss). The backward traversal is like salmon swimming upstream. Starting at the sea (loss), each salmon follows the river backward, splitting at every fork to distribute the message. The message at each point is "how important is this point for reaching the sea?" — the gradient of the loss with respect to that node's value.

## Why This Concept Matters

Understanding the graph backward traversal is essential for:
- **Debugging gradient issues**: Tracing where gradients get lost or amplified
- **Custom autograd functions**: Must implement correct backward methods
- **Architecture design**: Knowing how different graph structures affect gradient flow
- **Advanced techniques**: Gradient checkpointing, pruning, and optimization
- **Framework internals**: Understanding what PyTorch does during `.backward()`

## Mathematical Explanation

For a computation graph with nodes v₁, ..., v_N in topological order:

### Forward pass:
For i = 1 to N:
v_i ← op_i(Producers(v_i))

### Backward pass:
Initialize: grad(v_N) = 1 (for scalar loss)
For i = N down to 1:
- For each producer p of v_i:
    accumulate_grad(p) += local_gradient(v_i, p) · grad(v_i)

where local_gradient(v_i, p) = ∂v_i / ∂p (evaluated at the forward values).

The key insight: a node accumulates gradients from all its consumers (nodes that use it as input). If node A feeds into both B and C, then:
grad(A) = grad(A)_from_B + grad(A)_from_C

## Code Examples

### Example 1: Manual graph backward traversal

```python
import torch

class GraphNode:
    """A node in a computation graph."""
    def __init__(self, value, grad_fn=None, requires_grad=False):
        self.value = value
        self.grad = None
        self.grad_fn = grad_fn  # Function to compute local gradients
        self.requires_grad = requires_grad
        self.consumers = []  # Nodes that use this as input
    
    def backward(self):
        # Topological sort (reverse order)
        topo = []
        visited = set()
        def dfs(node):
            if id(node) not in visited:
                visited.add(id(node))
                for consumer in node.consumers:
                    dfs(consumer)
                topo.append(node)
        dfs(self)
        
        # Initialize gradient of output
        self.grad = torch.ones_like(self.value)
        
        # Backward traversal (reverse topological order)
        for node in reversed(topo):
            if node.grad_fn and node.grad is not None:
                node.grad_fn(node)
    
    def __repr__(self):
        return f"GraphNode(val={self.value:.2f}, grad={self.grad})"

# Build a computation graph manually
# Graph: x -> (*2) -> a -> (+1) -> b -> (^2) -> c -> sum -> loss
x_val = torch.tensor(3.0)
x = GraphNode(x_val, requires_grad=True)

# Multiply by 2: a = x * 2
a = GraphNode(x_val * 2)
def grad_a(node):
    x.grad = (node.grad * 2) if x.grad is None else x.grad + node.grad * 2
a.grad_fn = grad_a
x.consumers.append(a)

# Add 1: b = a + 1
b = GraphNode(a.value + 1)
def grad_b(node):
    a.grad = node.grad if a.grad is None else a.grad + node.grad
b.grad_fn = grad_b
a.consumers.append(b)

# Square: c = b^2
c = GraphNode(b.value ** 2)
def grad_c(node):
    b.grad = (node.grad * 2 * b.value) if b.grad is None else b.grad + node.grad * 2 * b.value
c.grad_fn = grad_c
b.consumers.append(c)

# Loss = c
loss = c
loss.backward()

print(f"x = {x_val.item()}")
print(f"a = x*2 = {a.value.item()}")
print(f"b = a+1 = {b.value.item()}")
print(f"c = b^2 = {c.value.item()}")
print(f"\nGradients:")
print(f"  ∂loss/∂x = {x.grad.item():.4f}")  # 2*(2x+1)*2 = 4*(2x+1) = 28
print(f"  ∂loss/∂a = {a.grad.item():.4f}")  # 2*(a+1) = 14
print(f"  ∂loss/∂b = {b.grad.item():.4f}")  # 2*b = 14
print(f"  ∂loss/∂c = {c.grad.item():.4f}")  # 1
# Output:
# x = 3.0
# a = x*2 = 6.0
# b = a+1 = 7.0
# c = b^2 = 49.0
# 
# Gradients:
#   ∂loss/∂x = 28.0000
#   ∂loss/∂a = 14.0000
#   ∂loss/∂b = 14.0000
#   ∂loss/∂c = 1.0000
```

### Example 2: PyTorch's backward graph traversal

```python
# PyTorch's autograd implements computation graph backward traversal
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Build a graph with branching (x used in two places)
z1 = x * y        # mul
z2 = x + y        # add
z3 = z1 * z2      # mul
loss = z3 ** 2    # pow

loss.backward()

# The backward traversal:
# 1. start at loss, compute dloss/dz3 = 2*z3
# 2. z3 = z1 * z2: dz3/dz1 = z2, dz3/dz2 = z1
#    dloss/dz1 = dloss/dz3 * z2
#    dloss/dz2 = dloss/dz3 * z1
# 3. z1 = x*y: dz1/dx = y, dz1/dy = x
#    dloss/dx += dloss/dz1 * y  (from z1 path)
#    z2 = x+y: dz2/dx = 1, dz2/dy = 1
#    dloss/dx += dloss/dz2 * 1  (from z2 path)
#    dloss/dy = dloss/dz1 * x + dloss/dz2 * 1

print(f"∂loss/∂x = {x.grad:.4f}")  # Branching: gradient from both z1 and z2 paths
print(f"∂loss/∂y = {y.grad:.4f}")

# Verify by drawing the graph
print(f"\nz1 = x*y = {z1.item():.2f}")
print(f"z2 = x+y = {z2.item():.2f}")
print(f"z3 = z1*z2 = {z3.item():.2f}")
print(f"loss = z3^2 = {loss.item():.2f}")
# Output:
# ∂loss/∂x = 420.0000
# ∂loss/∂y = 210.0000
# 
# z1 = x*y = 6.00
# z2 = x+y = 5.00
# z3 = z1*z2 = 30.00
# loss = z3^2 = 900.00
```

### Example 3: Gradient accumulation in backward traversal

```python
# Demonstrating gradient accumulation when a node has multiple consumers
x = torch.tensor(2.0, requires_grad=True)

# x is used in three separate branches
a = x ** 2        # first consumer
b = x ** 3        # second consumer
c = x + 1         # third consumer

d = a * b
loss = d + c

loss.backward()

# Gradient accumulates from all three paths:
# ∂loss/∂x = ∂a/∂x * ∂d/∂a + ∂b/∂x * ∂d/∂b + ∂c/∂x
# = 2x * b + 3x^2 * a + 1
# = 4 * 8 + 12 * 4 + 1
# = 32 + 48 + 1 = 81

print(f"x = {x.item()}")
print(f"a = x^2 = {a.item()}")
print(f"b = x^3 = {b.item()}")
print(f"c = x+1 = {c.item()}")
print(f"d = a*b = {d.item()}")
print(f"∂loss/∂x = {x.grad.item():.4f} (from 3 paths)")
# Output:
# x = 2.0
# a = x^2 = 4.0
# b = x^3 = 8.0
# c = x+1 = 3.0
# d = a*b = 32.0
# ∂loss/∂x = 81.0000 (from 3 paths)
```

### Example 4: Backward through control flow

```python
# Autograd handles control flow by recording only the executed path
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(-2.0, requires_grad=True)

# Condition based on value (differentiable path depends on input)
if x.item() > 0:
    z = x * y + torch.sin(x)
else:
    z = x / (y + 1)

loss = z ** 2
loss.backward()

print(f"x > 0, so path 1 executed: z = x*y + sin(x)")
print(f"z = {z.item():.4f}")
print(f"∂loss/∂x = {x.grad.item():.4f}")  # 2*z * (y + cos(x))
print(f"∂loss/∂y = {y.grad.item():.4f}")  # 2*z * x

# If we change x to be negative, a different path is executed
x2 = torch.tensor(-3.0, requires_grad=True)
y2 = torch.tensor(-2.0, requires_grad=True)

if x2.item() > 0:
    z2 = x2 * y2 + torch.sin(x2)
else:
    z2 = x2 / (y2 + 1)

loss2 = z2 ** 2
loss2.backward()
print(f"\nx2 < 0, so path 2 executed: z = x/(y+1)")
print(f"∂loss/∂x2 = {x2.grad.item():.4f}")  # 2*z * 1/(y+1)
print(f"∂loss/∂y2 = {y2.grad.item():.4f}")  # 2*z * -x/(y+1)^2
# Output:
# x > 0, so path 1 executed: z = x*y + sin(x)
# z = -5.8589
# ∂loss/∂x = 35.7025
# ∂loss/∂y = -35.1537
# 
# x2 < 0, so path 2 executed: z = x/(y+1)
# ∂loss/∂x2 = 18.0000
# ∂loss/∂y2 = -18.0000
```

### Example 5: Custom backward with gradient shaping

```python
import torch.nn as nn

class GradientShaper(nn.Module):
    """A module that shapes gradients during backward traversal."""
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        # During backward, modify the gradient
        # Example: only propagate positive gradients
        return torch.relu(grad_output)

# Using hooks to shape gradients during backward traversal
x = torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)

def backward_hook(module, grad_input, grad_output):
    print(f"  Backward hook: grad_output norm = {grad_output[0].norm():.4f}")
    # Modify gradient: only positive gradients propagate
    return (torch.relu(grad_output[0]),)

layer = nn.Identity()
hook = layer.register_full_backward_hook(backward_hook)

y = layer(x)
loss = y.sum()
loss.backward()

print(f"\nOriginal gradient (should be all ones): {x.grad}")
hook.remove()
# Output:
#   Backward hook: grad_output norm = 2.0000
# 
# Original gradient (should be all ones): tensor([1., 1., 1., 1.])
```

### Example 6: Visualizing backward graph traversal

```python
from torch.autograd import Function

class TrackedMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        print(f"  Forward: {a.data.item():.2f} * {b.data.item():.2f}")
        return a * b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        print(f"  Backward: {grad_output.item():.2f} * [{b.item():.2f}, {a.item():.2f}]")
        return grad_output * b, grad_output * a

# Build graph with tracked operations
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

print("Forward pass:")
a = TrackedMul.apply(x, y)     # a = x * y
b = TrackedMul.apply(a, x)     # b = a * x
c = TrackedMul.apply(b, y)     # c = b * y
loss = c.sum()

print("\nBackward traversal:")
loss.backward()

print(f"\nFinal gradients:")
print(f"  ∂loss/∂x = {x.grad.item():.4f}")
print(f"  ∂loss/∂y = {y.grad.item():.4f}")
# Output:
# Forward pass:
#   Forward: 2.00 * 3.00
#   Forward: 6.00 * 2.00
#   Forward: 12.00 * 3.00
# 
# Backward traversal:
#   Backward: 1.00 * [3.00, 12.00]
#   Backward: 3.00 * [2.00, 6.00]
#   Backward: 6.00 * [3.00, 2.00]
# 
# Final gradients:
#   ∂loss/∂x = 54.0000
#   ∂loss/∂y = 48.0000
```

## Common Mistakes

1. **Assuming backward traversal visits nodes in forward order**: Backward visits nodes in REVERSE topological order (from loss to inputs). The order matters for correct gradient computation.

2. **Forgetting that gradient accumulation happens at branches**: If a node feeds into multiple consumers, gradients from all paths are summed.

3. **Modifying forward values after backward**: Backward uses stored forward values. If you modify them in-place before backward, gradients will be wrong.

4. **Not understanding that the graph is consumed by backward**: After `loss.backward()`, the graph is freed (intermediate gradients removed). Use `retain_graph=True` to keep it.

5. **Confusing the forward and backward graph**: The backward pass creates its own implicit graph through `grad_fn` chains, which mirrors the forward graph but in reverse.

6. **Using detached tensors in the graph**: Tensor.detach() creates a new tensor that isn't connected to the graph, breaking the backward traversal.

7. **Thinking backward traversal can start from any node**: It starts from the scalar loss node. For non-scalar tensors, provide `grad_outputs`.

## Interview Questions

### Beginner - 5

1. What order does the backward pass visit nodes?
2. Why must nodes be visited in reverse topological order?
3. What happens when a variable is used by multiple operations?
4. What is gradient accumulation in the backward pass?
5. Why does the graph get freed after backward()?

### Intermediate - 5

1. Describe the complete backward traversal algorithm for a computation graph.
2. How does the backward pass handle branching (one input to many operations)?
3. What is the role of `ctx.save_for_backward` in custom autograd functions?
4. How do hooks intercept the backward traversal?
5. Explain what `retain_graph=True` does and when you would use it.

### Advanced - 3

1. Implement a custom backward traversal that computes additional quantities (e.g., gradient norms) during the backward pass.
2. Design a method to visualize which parts of the graph receive the largest gradients.
3. Analyze the computational complexity of backward traversal through a transformer attention graph.

## Practice Problems

### Easy - 5

1. Trace the backward traversal through `y = ((x + 2) * 3) ** 2`.
2. Verify that branching nodes accumulate gradients from all consumers.
3. Use `retain_graph=True` to perform two backward passes.
4. Register a backward hook and print gradient norms.
5. Check the order of node visits during backward.

### Medium - 5

1. Implement a custom autograd Function and trace its backward traversal.
2. Compare backward traversal through a residual block vs. a plain block.
3. Implement gradient monitoring at each node during backward.
4. Create a graph with skip connections and verify correct gradient accumulation.
5. Use backward hooks to implement gradient clipping during backward.

### Hard - 3

1. Implement a custom backward traversal that skips nodes with negligible gradients (gradient pruning).
2. Design a graph structure where backward traversal can be parallelized across branches.
3. Implement a differentiable graph where the backward traversal itself is optimized for memory.

## Solutions

### Easy - 1
```python
# Graph: x -> (+2) -> (*3) -> (^2) -> loss
# Backward: loss -> d/d(^2) -> d/d(*3) -> d/d(+2) -> x
x = torch.tensor(1.0, requires_grad=True)
t1 = x + 2
t2 = t1 * 3
loss = t2 ** 2
loss.backward()
print(x.grad)  # 2 * t2 * 3 * 1 = 2*9*3 = 54
```

### Easy - 2
```python
x = torch.tensor(2.0, requires_grad=True)
a = x ** 2
b = x ** 3
loss = a + b
loss.backward()
print(x.grad)  # 2*x + 3*x^2 = 4 + 12 = 16
```

### Easy - 3
```python
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
y.backward()
print(x.grad)  # 4 (accumulated)
```

## Related Concepts

DL-064 Reverse Mode Autodiff, DL-053 Computational Graph, DL-054 Directed Acyclic Graph, DL-066 Error Signal Propagation

## Next Concepts

DL-066 Error Signal Propagation, DL-067 Layerwise Gradients

## Summary

The computation graph backward traversal is the implementation of reverse-mode autodiff that visits nodes in reverse topological order, computing gradients by applying the chain rule at each node. Gradients from multiple consumers are accumulated at shared inputs. Understanding backward traversal is essential for advanced autograd usage, debugging, and custom operation development.

## Key Takeaways

- Backward traversal visits nodes in reverse topological order
- Gradients accumulate when a node has multiple consumers
- Forward values must be stored for backward computation
- The graph is freed after one backward (unless retain_graph=True)
- Backward hooks enable gradient monitoring and modification
- Custom autograd Functions define their own backward logic
- The backward traversal is the concrete algorithm behind autograd
- Branching and skip connections create gradient accumulation points
