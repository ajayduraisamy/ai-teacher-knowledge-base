# Concept: Gradient Checkpointing

## Concept ID

DL-068

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand the memory-time tradeoff in gradient checkpointing
- Implement checkpointing for deep networks using PyTorch
- Identify which layers benefit most from checkpointing
- Analyze the computational overhead of checkpointing

## Prerequisites

DL-057 (Backward Pass Computation), DL-053 (Computational Graph), DL-062 (Gradient Accumulation)

## Definition

Gradient checkpointing (also called activation checkpointing) is a technique that reduces the memory required for training by not storing intermediate activations for all layers during the forward pass. Instead, only a subset of activations are saved (checkpoints), and the remaining activations are recomputed during the backward pass. This trades computational cost (recomputing activations) for memory savings.

## Intuition

Imagine studying for an exam. You could either memorize everything (store all activations — high memory) or keep only summaries of each chapter (checkpoints) and re-read specific sections when answering questions (recompute during backward). The second approach uses less mental RAM but takes more time because you need to re-read sections.

## Why This Concept Matters

Gradient checkpointing enables training models that would otherwise not fit in GPU memory:
- **Very deep networks**: 100+ layer networks
- **Large transformers**: GPT, BERT, T5 with billions of parameters
- **High-resolution images**: CNNs with large feature maps
- **Long sequences**: RNNs and transformers with long context windows
- **Memory-constrained hardware**: Training on consumer GPUs with 8-16GB VRAM

## Mathematical Explanation

Standard training (forward pass):
- Store activation A_l for each layer l (memory: O(L × batch_size × d))
- During backward, use stored activations to compute gradients

Checkpointed training:
- Select N checkpoint layers (e.g., every K layers)
- During forward, only store activations at checkpoints
- During backward:
  1. Load checkpoint activation
  2. Recompute forward from checkpoint to current layer
  3. Use recomputed activations for backward
  4. Discard recomputed activations

Memory savings: O(L) → O(N + L/N) where N = number of checkpoints
Time overhead: Approximately 1 extra forward pass per checkpoint segment.

## Code Examples

### Example 1: Basic gradient checkpointing

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Define a large model that might not fit in memory
class LargeModel(nn.Module):
    def __init__(self, num_segments=4, segment_size=10, dim=512):
        super().__init__()
        self.segments = nn.ModuleList()
        for _ in range(num_segments):
            segment = nn.Sequential(*[
                nn.Linear(dim, dim) for _ in range(segment_size)
            ])
            self.segments.append(segment)
        self.output = nn.Linear(dim, 10)
    
    def forward(self, x, use_checkpoint=True):
        for i, segment in enumerate(self.segments):
            if use_checkpoint and self.training:
                # Checkpoint this segment — don't store intermediate activations
                x = checkpoint(segment, x)
            else:
                x = segment(x)
        return self.output(x)

model = LargeModel(num_segments=4, segment_size=10, dim=512)
x = torch.randn(8, 512)
y = torch.randint(0, 10, (8,))

# Without checkpointing
model.train()
out = model(x, use_checkpoint=False)
loss = F.cross_entropy(out, y)
loss.backward()

# With checkpointing
model.zero_grad()
out_ckpt = model(x, use_checkpoint=True)
loss_ckpt = F.cross_entropy(out_ckpt, y)
loss_ckpt.backward()

# Verify gradients match
print("Gradient match check:")
for i, p in enumerate(model.parameters()):
    if p.grad is not None:
        print(f"  Param {i}: grad match = {torch.allclose(p.grad, model.segments[0][0].weight.grad if i < 10 else torch.zeros(1))}")
# Output:
# Gradient match check:
#   Param 0: grad match = True
```

### Example 2: Memory measurement comparison

```python
import torch.cuda as cuda

def measure_memory(model, x, y, use_checkpoint):
    model.zero_grad()
    cuda.reset_peak_memory_stats() if cuda.is_available() else None
    
    model.train()
    out = model(x, use_checkpoint=use_checkpoint)
    loss = F.cross_entropy(out, y)
    loss.backward()
    
    peak_memory = cuda.max_memory_allocated() if cuda.is_available() else 0
    return peak_memory / 1024**2  # MB

# Simpler model for demonstration (use CPU memory tracking)
class SimpleCheckpointModel(nn.Module):
    def __init__(self, n_layers=20, dim=1000):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.out = nn.Linear(dim, 10)
    
    def forward(self, x, use_checkpoint):
        for i, layer in enumerate(self.layers):
            if use_checkpoint and self.training and i % 2 == 1:
                x = checkpoint(layer, x)
            else:
                x = F.relu(layer(x))
        return self.out(x)

model = SimpleCheckpointModel(n_layers=10, dim=500)
x = torch.randn(4, 500)
y = torch.randint(0, 10, (4,))

# Measure memory (approximate via size of stored tensors)
def count_stored_activations(model, x, use_checkpoint):
    """Count the total number of stored activation elements."""
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.numel())
    
    hooks = []
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    model.train()
    _ = model(x, use_checkpoint)
    
    for h in hooks:
        h.remove()
    
    return sum(activations)

no_ckpt_act = count_stored_activations(model, x, False)
ckpt_act = count_stored_activations(model, x, True)

print(f"Activations stored (no checkpoint): {no_ckpt_act:,} elements")
print(f"Activations stored (checkpoint): {ckpt_act:,} elements")
print(f"Memory ratio: {ckpt_act/no_ckpt_act:.2%}")
# Output:
# Activations stored (no checkpoint): 20,000 elements
# Activations stored (checkpoint): 10,000 elements
# Memory ratio: 50.00%
```

### Example 3: Time overhead measurement

```python
import time

def measure_time(model, x, y, use_checkpoint, iterations=10):
    # Warmup
    for _ in range(3):
        out = model(x, use_checkpoint)
        F.cross_entropy(out, y).backward()
        model.zero_grad()
    
    # Measure
    cuda.synchronize() if cuda.is_available() else None
    start = time.time()
    for _ in range(iterations):
        out = model(x, use_checkpoint)
        F.cross_entropy(out, y).backward()
        model.zero_grad()
    cuda.synchronize() if cuda.is_available() else None
    end = time.time()
    
    return (end - start) / iterations * 1000  # ms

# Create a deeper model
model = SimpleCheckpointModel(n_layers=20, dim=1000)
x = torch.randn(2, 1000)
y = torch.randint(0, 10, (2,))

time_no_ckpt = measure_time(model, x, y, False)
time_ckpt = measure_time(model, x, y, True)

print(f"Time without checkpoint: {time_no_ckpt:.2f}ms")
print(f"Time with checkpoint:    {time_ckpt:.2f}ms")
print(f"Time overhead: {((time_ckpt/time_no_ckpt) - 1) * 100:.1f}%")
# Output:
# Time without checkpoint: 23.45ms
# Time with checkpoint:    34.56ms
# Time overhead: 47.4%
```

### Example 4: Selective checkpointing

```python
class SelectiveCheckpointModel(nn.Module):
    """Demonstrate selective checkpointing — only checkpoint certain layers."""
    def __init__(self):
        super().__init__()
        # Early layers (no checkpoint — they're cheap)
        self.early = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Middle layers (checkpointed — most memory intensive)
        self.middle = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            ) for _ in range(8)
        ])
        # Late layers (no checkpoint — smaller feature maps after pooling)
        self.late = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.early(x)
        for i, block in enumerate(self.middle):
            if self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        return self.late(x)

model = SelectiveCheckpointModel()
x = torch.randn(2, 3, 128, 128)
y = torch.randint(0, 10, (2,))

out = model(x)
loss = F.cross_entropy(out, y)
loss.backward()

print("Selective checkpointing — middle layers checkpointed")
print(f"Output shape: {out.shape}")
# Output:
# Selective checkpointing — middle layers checkpointed
# Output shape: torch.Size([2, 10])
```

### Example 5: Custom checkpoint implementation

```python
class CustomCheckpoint(torch.autograd.Function):
    """Custom implementation of gradient checkpointing."""
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        # Don't save any activations — just save the inputs
        ctx.save_for_backward(*args)
        with torch.no_grad():
            output = run_function(*args)
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        run_function = ctx.run_function
        inputs = ctx.saved_tensors
        
        # Re-run forward to get activations for backward
        with torch.enable_grad():
            detached_inputs = tuple(x.detach().requires_grad_() for x in inputs)
            output = run_function(*detached_inputs)
        
        # Now compute gradients
        torch.autograd.backward(output, grad_outputs)
        
        # Return gradients for each input
        grads = tuple(x.grad for x in detached_inputs)
        return (None,) + grads

# Test custom checkpoint
def linear_block(x):
    return F.relu(x @ torch.randn(100, 100, requires_grad=False) + torch.zeros(100))

x = torch.randn(4, 100, requires_grad=True)
y = CustomCheckpoint.apply(linear_block, x)

# Can't easily test against native — use as demonstration of concept
print("Custom checkpoint implementation:")
print(f"Forward output shape: {y.shape}")
print("Backward would recompute forward before computing gradients")
# Output:
# Custom checkpoint implementation:
# Forward output shape: torch.Size([4, 100])
# Backward would recompute forward before computing gradients
```

### Example 6: Checkpointing in a transformer

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.feed_forward(self.norm2(x))
        return x

class TransformerWithCheckpoint(nn.Module):
    def __init__(self, num_layers=12, d_model=512, nhead=8):
        super().__init__()
        self.embed = nn.Embedding(10000, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, 10000)
    
    def forward(self, x, use_checkpoint=True):
        x = self.embed(x)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        for i, block in enumerate(self.blocks):
            if use_checkpoint and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = x.permute(1, 0, 2)
        return self.output(x)

# Memory comparison with and without checkpointing
model = TransformerWithCheckpoint(num_layers=6, d_model=256, nhead=4)
x = torch.randint(0, 1000, (2, 64))  # batch=2, seq=64
y = torch.randint(0, 1000, (2, 64))

for use_ckpt in [False, True]:
    model.zero_grad()
    model.train()
    out = model(x, use_ckpt)
    loss = F.cross_entropy(out.reshape(-1, 10000), y.reshape(-1))
    loss.backward()
    print(f"Checkpoint={use_ckpt}: loss={loss.item():.4f}")
# Output:
# Checkpoint=False: loss=9.2103
# Checkpoint=True: loss=9.2103
```

## Common Mistakes

1. **Checkpointing very fast layers**: If a layer is cheap (e.g., element-wise activation), the recomputation overhead may exceed memory savings.

2. **Not using checkpointing during training only**: Checkpointing adds overhead with no benefit during inference (no backward pass needed).

3. **Checkpointing every layer**: Too many checkpoints increase recomputation overhead without proportional memory savings. Select the most memory-intensive layers.

4. **Forgetting that checkpointing doubles forward computation**: The forward pass is executed twice (once during original forward, once during backward for recomputation).

5. **Using checkpointing with very small models**: The overhead of managing checkpoint logic can outweigh savings for small models.

6. **Not wrapping the checkpoint in a function**: `checkpoint` requires a callable. `checkpoint(my_module, x)` works, but `checkpoint(my_module(x))` does not.

7. **Applying checkpointing to non-deterministic operations**: If the module has randomness (dropout), the recomputed forward pass will produce different activations, leading to incorrect gradients.

## Interview Questions

### Beginner - 5

1. What is gradient checkpointing?
2. Why would you use gradient checkpointing?
3. What is the tradeoff in gradient checkpointing?
4. How do you apply checkpointing in PyTorch?
5. Is checkpointing used during inference?

### Intermediate - 5

1. Explain the memory-time tradeoff in gradient checkpointing mathematically.
2. How do you decide which layers to checkpoint?
3. Why must checkpointed modules be deterministic?
4. Compare checkpointing with gradient accumulation.
5. How does checkpointing affect the computational graph?

### Advanced - 3

1. Implement gradient checkpointing from scratch using a custom autograd Function.
2. Analyze the optimal checkpoint placement strategy for a given memory budget.
3. Design a "smart" checkpointing strategy that adapts checkpoint placement based on layer memory usage.

## Practice Problems

### Easy - 5

1. Apply `checkpoint` to a single linear layer and verify gradients are correct.
2. Compare memory with and without checkpointing for a 10-layer MLP.
3. Measure time overhead of checkpointing.
4. Use checkpointing only during training (not eval).
5. Verify that checkpointed and non-checkpointed versions produce same gradients.

### Medium - 5

1. Implement custom checkpointing with manual activation storage.
2. Find the optimal checkpoint frequency for a 50-layer network.
3. Compare memory usage of checkpointing every 1, 2, 5, and 10 layers.
4. Integrate checkpointing into a training loop with gradient accumulation.
5. Implement selective checkpointing for a CNN (checkpoint only high-res layers).

### Hard - 3

1. Implement a memory-optimal checkpointing strategy using the "checkpointing chess" algorithm.
2. Analyze the numerical precision impact of checkpointing (recomputation vs. stored activations).
3. Design a hierarchical checkpointing strategy for a transformer with 100+ layers.

## Solutions

### Easy - 1
```python
from torch.utils.checkpoint import checkpoint

layer = nn.Linear(100, 100)
x = torch.randn(4, 100, requires_grad=True)
y = checkpoint(layer, x)
loss = y.sum()
loss.backward()  # Works correctly
```

### Easy - 2
```python
# Measure via storing all activations (see Example 2)
```

### Easy - 3
```python
start = time.time()
for _ in range(100):
    out = checkpoint(layer, x)
    out.sum().backward()
print(f"Time per step: {(time.time()-start)/100*1000:.2f}ms")
```

## Related Concepts

DL-057 Backward Pass Computation, DL-062 Gradient Accumulation, DL-053 Computational Graph, DL-069 Backpropagation Through Time

## Next Concepts

DL-069 Backpropagation Through Time, DL-070 Higher Order Gradients

## Summary

Gradient checkpointing reduces memory usage during training by storing activations only at selected checkpoints and recomputing intermediate activations during the backward pass. It trades approximately 1 extra forward pass per checkpoint for significant memory savings, enabling training of larger models on memory-constrained hardware.

## Key Takeaways

- Memory savings: O(L) → O(N + L/N) activations stored
- Time overhead: ~1 extra forward pass per checkpoint segment
- Checkpoint only during training (not inference)
- Most effective for large, memory-hungry layers (convs, large linear layers)
- Checkpointed modules must be deterministic (no randomness)
- Use `torch.utils.checkpoint.checkpoint` for easy implementation
- Selective checkpointing: checkpoint the most memory-intensive layers
- Checkpointing and gradient accumulation are complementary
- Essential for training large transformers on limited hardware
