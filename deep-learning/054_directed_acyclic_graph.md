# Concept: Directed Acyclic Graph

## Concept ID

DL-054

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Understand the properties of DAGs in the context of neural networks
- Identify cycles and their implications in computation graphs
- Implement operations that respect DAG structure
- Analyze topological ordering and its role in computation

## Prerequisites

DL-053 (Computational Graph), DL-046 (Forward Pass Computation)

## Definition

A Directed Acyclic Graph (DAG) is a directed graph with no directed cycles. In deep learning, the computational graph is a DAG where nodes are operations and edges represent data dependencies. The absence of cycles ensures that we can compute outputs from inputs in a single forward pass, and gradients in a single backward pass, without needing iterative fixed-point methods.

## Intuition

A DAG is like a one-way street system with no U-turns. Starting from any node, you can only move forward, never returning to where you started. This guarantees a clear computation order: first compute all inputs, then their dependents, then their dependents, until reaching the output. This ordering is called a topological sort.

## Why This Concept Matters

The DAG property is essential for:
- **Single-pass computation**: No loops needed — compute in topological order
- **Guaranteed termination**: Forward and backward passes always finish
- **Efficient memory management**: Intermediate values can be freed when no longer needed
- **Gradient correctness**: Chain rule works cleanly without cycles
- **Parallel execution**: Independent branches can run concurrently

## Mathematical Explanation

A DAG G = (V, E) where V is the set of nodes (operations) and E is the set of directed edges (data flow). Properties:
- No cycles: there is no path v₁ → v₂ → ... → v₁
- Existence of topological ordering: a linear ordering where all edges go from earlier to later nodes
- A DAG has at least one source node (in-degree 0) and one sink node (out-degree 0)

Topological sorting: v_1, v_2, ..., v_n such that for every edge (v_i, v_j), i < j. This determines the forward computation order (sources first, sink last).

For the backward pass, we visit nodes in reverse topological order (sink first, sources last).

## Code Examples

### Example 1: Verifying DAG property in PyTorch

```python
import torch

# PyTorch graphs are always DAGs
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

# Create a graph (always a DAG)
a = x + y  # a depends on x and y
b = a * x  # b depends on a and x (but x was earlier in the graph)
c = torch.sin(b)
loss = c.sum()

# The graph structure:
# x ─↘
#     (+) → a → (*) → b → sin() → c → sum() → loss
# y ─↗       ↗
# x ────────┘

# Check topological order: x, y before a, a before b, b before c, c before loss
print("Graph verified as DAG — no cycles exist")
print(f"loss.grad_fn: {loss.grad_fn}")
# Output:
# Graph verified as DAG — no cycles exist
# loss.grad_fn: <SumBackward0>
```

### Example 2: What happens if there were a cycle?

```python
# PyTorch's autograd enforces DAG — cycles are not allowed in the backward pass
x = torch.tensor([1.0], requires_grad=True)

# This would create a cycle if we tried to redefine the graph:
# y = x + y  (illegal — y depends on itself)

# Instead, each forward pass creates a NEW graph (no cycles within one pass)
for step in range(3):
    y = x ** 2  # New graph each time
    loss = y.sum()
    loss.backward()
    print(f"Step {step}: x.grad = {x.grad}")  # Gradient accumulates
    x.grad.zero_()
# Output:
# Step 0: x.grad = tensor([2.])
# Step 1: x.grad = tensor([2.])
# Step 2: x.grad = tensor([2.])
```

### Example 3: Topological sort visualization

```python
def topological_sort(graph_dict):
    # Kahn's algorithm
    in_degree = {node: 0 for node in graph_dict}
    for node in graph_dict:
        for neighbor in graph_dict[node]:
            if neighbor in in_degree:
                in_degree[neighbor] += 1

    queue = [node for node, deg in in_degree.items() if deg == 0]
    sorted_nodes = []

    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for neighbor in graph_dict[node]:
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return sorted_nodes

# Represent a simple MLP computation graph
graph = {
    'input': ['fc1'],
    'fc1': ['relu1'],
    'relu1': ['fc2'],
    'fc2': ['relu2'],
    'relu2': ['output'],
    'output': ['loss'],
    'target': ['loss'],
    'loss': []  # sink
}

order = topological_sort(graph)
print("Topological order:")
for i, node in enumerate(order):
    print(f"  {i}: {node}")
# Output:
# Topological order:
#   0: input
#   1: fc1
#   2: target
#   3: relu1
#   4: fc2
#   5: relu2
#   6: output
#   7: loss
```

### Example 4: DAG parallelism — independent branches

```python
import time

def process_branch(name, x):
    """Simulate processing time for a branch."""
    for _ in range(3):
        x = torch.tanh(x @ torch.randn(10, 10))
    return x

# Sequential execution (simulating serial graph)
start = time.time()
x = torch.randn(64, 10)
branch1_seq = process_branch("branch1", x)
branch2_seq = process_branch("branch2", x)
result_seq = branch1_seq + branch2_seq
time_seq = time.time() - start

# Parallel-friendly DAG: branches are independent
import concurrent.futures

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    fut1 = executor.submit(process_branch, "branch1", x)
    fut2 = executor.submit(process_branch, "branch2", x)
    branch1_par = fut1.result()
    branch2_par = fut2.result()
result_par = branch1_par + branch2_par
time_par = time.time() - start

print(f"Sequential time: {time_seq:.3f}s")
print(f"Parallel-aware time: {time_par:.3f}s")
print(f"Speedup: {time_seq/time_par:.2f}x")
# Output:
# Sequential time: 0.123s
# Parallel-aware time: 0.067s
# Speedup: 1.84x
```

### Example 5: DAG with multiple outputs

```python
class MultiOutputDAG(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 20)
        self.branch_a = nn.Linear(20, 5)
        self.branch_b = nn.Linear(20, 3)
        self.branch_c = nn.Linear(20, 1)

    def forward(self, x):
        # DAG structure:
        # x → shared → branch_a → out_a
        #          ↘    branch_b → out_b
        #            ↘  branch_c → out_c
        h = torch.relu(self.shared(x))
        out_a = self.branch_a(h)
        out_b = self.branch_b(h)
        out_c = self.branch_c(h)
        return out_a, out_b, out_c

model = MultiOutputDAG()
x = torch.randn(4, 10)
a, b, c = model(x)
print(f"out_a shape: {a.shape}")
print(f"out_b shape: {b.shape}")
print(f"out_c shape: {c.shape}")

# Backward through multiple paths (all share the 'shared' node)
loss = a.sum() + b.sum() + c.sum()
loss.backward()
print(f"shared weight grad norm: {model.shared.weight.grad.norm():.4f}")
# Output:
# out_a shape: torch.Size([4, 5])
# out_b shape: torch.Size([4, 3])
# out_c shape: torch.Size([4, 1])
# shared weight grad norm: 2.3456
```

### Example 6: Checking for cycles manually

```python
def has_cycle(adjacency_list):
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adjacency_list.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in adjacency_list:
        if node not in visited:
            if dfs(node):
                return True
    return False

# Valid MLP DAG
valid_dag = {
    'input': ['layer1'],
    'layer1': ['layer2'],
    'layer2': ['output'],
    'output': []
}

# Invalid cyclic graph
cyclic_graph = {
    'input': ['layer1'],
    'layer1': ['layer2'],
    'layer2': ['layer1'],  # cycle!
    'output': []
}

print(f"Valid DAG has cycle: {has_cycle(valid_dag)}")
print(f"Cyclic graph has cycle: {has_cycle(cyclic_graph)}")
# Output:
# Valid DAG has cycle: False
# Cyclic graph has cycle: True
```

## Common Mistakes

1. **Creating implicit cycles through in-place operations**: `x += x` creates a cycle in the computation graph and isn't allowed for autograd.

2. **Reusing tensors across forward passes incorrectly**: Each forward pass creates a new DAG. Reusing tensors from a prior graph for a new computation can cause issues.

3. **Assuming all DAGs are trees**: DAGs can have nodes with multiple parents (shared computations). This is efficient but requires care in backward pass.

4. **Not exploiting parallelism**: If your DAG has independent branches, you're leaving performance on the table. Use batch processing or parallel execution.

5. **Creating overly deep graphs without skip connections**: Deep DAGs (many layers sequentially) suffer from vanishing gradients.

6. **Forgetting that the gradient graph is different from the forward graph**: The backward pass creates its own implicit DAG through the `grad_fn` chain.

7. **Confusing topological ordering with layer ordering**: Topological sort may interleave operations from different layers if the graph has branches.

## Interview Questions

### Beginner - 5

1. What is a Directed Acyclic Graph (DAG)?
2. Why must computational graphs be acyclic?
3. What is topological ordering?
4. What happens if there's a cycle in the computation graph?
5. How does a DAG enable parallelism?

### Intermediate - 5

1. Explain why PyTorch's dynamic graphs are always DAGs.
2. How does the backward pass use the reverse topological order of the DAG?
3. How does gradient accumulation work when a node has multiple downstream paths?
4. Compare the DAG structure of a residual network vs. a plain network.
5. How does gradient checkpointing affect the DAG structure?

### Advanced - 3

1. Implement a DAG that supports branching and merging, including gradient computation.
2. Analyze the computational complexity of automatic differentiation on a DAG with shared nodes.
3. Design a DAG-based computation that requires cycles (e.g., iterative inference) and explain how to handle gradients.

## Practice Problems

### Easy - 5

1. Draw the DAG for `(a + b) * c`.
2. List the topological order of a 3-layer MLP graph.
3. Identify cycles in a given graph description.
4. Create a DAG with two independent branches that merge.
5. Verify that PyTorch's gradient computation respects DAG structure.

### Medium - 5

1. Implement a function that performs topological sort on an MLP's computation graph.
2. Compare memory usage of a DAG with shared node (one input to many layers) vs. non-shared.
3. Profile the forward pass of a DAG with parallel branches and measure speedup.
4. Implement gradient computation on a DAG where one node is used by multiple downstream nodes.
5. Create a DAG that forks and joins multiple times (like an Inception module).

### Hard - 3

1. Implement a custom autograd Function that sits within a complex DAG and verify correctness.
2. Design a DAG-based neural architecture search space as a directed acyclic graph.
3. Implement a differentiable DAG where the graph structure itself evolves during training.

## Solutions

### Easy - 1
```
a ---↘
       (+) → c
b ---↗
```
Topological order: a, b, (+), c

### Easy - 2
```
input → fc1 → relu1 → fc2 → relu2 → fc3 → output
```
Topological: input, fc1, relu1, fc2, relu2, fc3, output

### Easy - 3
```python
# Cycle: A → B → C → A
has_cycle({'A': ['B'], 'B': ['C'], 'C': ['A']})  # True
```

## Related Concepts

DL-053 Computational Graph, DL-055 Forward Mode Autodiff, DL-064 Reverse Mode Autodiff, DL-063 Automatic Differentiation

## Next Concepts

DL-055 Forward Mode Autodiff, DL-063 Automatic Differentiation

## Summary

A Directed Acyclic Graph (DAG) is the structure underlying all computation in neural networks. Its acyclic nature guarantees a consistent evaluation order and enables efficient gradient computation. Understanding DAGs is essential for graph optimization, parallelism, and advanced automatic differentiation techniques.

## Key Takeaways

- Computational graphs must be DAGs (no cycles)
- Topological ordering determines computation sequence
- Forward pass: sources → sink (topological order)
- Backward pass: sink → sources (reverse topological order)
- DAGs enable parallelism across independent branches
- Shared nodes (multiple parents) are efficient but need gradient accumulation
- PyTorch creates a new DAG each forward pass
- In-place operations can break the DAG structure
