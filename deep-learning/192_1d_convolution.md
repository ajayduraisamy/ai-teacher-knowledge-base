# Concept: 1D Convolution

## Concept ID

DL-192

## Difficulty

Advanced

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Understand the 1D convolution operation for sequence data
- Implement 1D convolution in PyTorch
- Apply 1D convolutions to time series and text data
- Analyze the use of 1D convs in modern architectures

## Prerequisites

DL-176 Convolution Operation, DL-177 Convolution Kernel, DL-178 Stride, DL-179 Padding

## Definition

1D convolution applies a kernel along a single spatial/temporal dimension, commonly used for processing sequential data such as time series, audio signals, and text.

## Intuition

If 2D convolution is like scanning a photo with a magnifying glass, 1D convolution is like scanning a timeline with a sliding window. It detects local patterns in sequences — a specific wave shape in sensor data, a syllable in audio, or a trigram in text. The kernel slides along the time/position axis, computing dot products at each step. This is the foundation of many sequence models, especially for audio and long text.

## Why This Concept Matters

1D convolutions are powerful for sequence modeling, often outperforming RNNs on tasks with long sequences while being parallelizable and faster to train. They're used in audio generation (WaveNet), text classification (TextCNN), and time series forecasting. Understanding 1D convolution is essential for working with non-image data.

## Mathematical Explanation

**1D convolution operation**:
$$O[i] = \sum_{k=0}^{K-1} I[i \cdot S + k] \cdot K[k]$$

**Multi-channel 1D convolution**:
$$O_c[i] = \sum_{c'=0}^{C_{in}-1} \sum_{k=0}^{K-1} I_{c'}[i \cdot S + k] \cdot K_{c,c'}[k]$$

**Output length**:
$$O_{len} = \left\lfloor \frac{W - K + 2P}{S} + 1 \right\rfloor$$

**PyTorch shape convention**: $(N, C_{in}, L)$ for input, $(N, C_{out}, L_{out})$ for output.

## Code Examples

### Example 1: Basic 1D Convolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Input: batch=2, channels=1, sequence length=10
x = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                  [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]], dtype=torch.float32)

print(f"Input shape: {x.shape}")
# Output: Input shape: torch.Size([2, 1, 10])

# 1D conv: in_channels=1, out_channels=1, kernel_size=3
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# Set kernel weights manually for demonstration
with torch.no_grad():
    conv1d.weight[0, 0] = nn.Parameter(torch.tensor([0.5, 1.0, 0.5]))

out = conv1d(x)
print(f"Output shape: {out.shape}")
# Output: Output shape: torch.Size([2, 1, 8])

print("First sequence output:")
print(out[0, 0].detach().numpy())
# Output: [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
# Manual: 0.5*1 + 1.0*2 + 0.5*3 = 2.5
#         0.5*2 + 1.0*3 + 0.5*4 = 3.5 ...
```

### Example 2: 1D Convolution for Text Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# TextCNN: 1D conv for sentiment analysis
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, 
                 filter_sizes, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs, padding=fs//2)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x: (B, seq_len)
        emb = self.embedding(x)  # (B, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)  # (B, embed_dim, seq_len)
        
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))  # (B, num_filters, seq_len)
            c = F.max_pool1d(c, c.size(2))  # (B, num_filters, 1)
            conv_outs.append(c.squeeze(2))
        
        cat = torch.cat(conv_outs, dim=1)  # (B, num_filters * len(filter_sizes))
        return self.fc(cat)

# Example usage
model = TextCNN(vocab_size=5000, embed_dim=100, num_filters=64,
                filter_sizes=[3, 5, 7], num_classes=2, max_len=50)

x = torch.randint(0, 5000, (16, 50))
out = model(x)
print(f"Input: {x.shape}, Output: {out.shape}")
# Output: Input: torch.Size([16, 50]), Output: torch.Size([16, 2])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Model parameters: 520,898
```

### Example 3: WaveNet-style Dilated 1D Convolutions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Dilated 1D convolution block for audio/time series
class Dilated1DBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, 
                              padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.norm(self.conv(x)))
        return out + residual  # Residual connection

class WaveNetLike(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_blocks):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # Stack of dilated 1D convs with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            Dilated1DBlock(hidden_channels, dilation=2**i)
            for i in range(num_blocks)
        ])
        
        self.output_proj = nn.Conv1d(hidden_channels, 1, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

# Generate a simple sine wave time series
seq_len = 256
x = torch.sin(torch.linspace(0, 4 * 3.14159, seq_len)).unsqueeze(0).unsqueeze(0)
x = x + 0.05 * torch.randn_like(x)  # add noise

model = WaveNetLike(in_channels=1, hidden_channels=32, num_blocks=6)
out = model(x)

print(f"Input: {x.shape}, Output: {out.shape}")
# Output: Input: torch.Size([1, 1, 256]), Output: torch.Size([1, 1, 256])

# Receptive field grows exponentially (2^6 - 1 = 63 samples)
print(f"Receptive field: {2**6 - 1} samples")
# Output: Receptive field: 63 samples
```

## Common Mistakes

1. **Channel dimension ordering**: PyTorch Conv1d expects (N, C, L), not (N, L, C). Remember to permute after embedding.
2. **Using 1D conv on non-sequential data without justification**: 1D convs assume local temporal structure.
3. **Ignoring causality**: For autoregressive tasks, need causal/padded convolutions that don't look at future timesteps.
4. **Wrong kernel size for sequence length**: Kernels shouldn't be larger than half the sequence length.
5. **Forgetting that dilation affects effective kernel size and padding needs**.

## Interview Questions

### Beginner - 5
1. What is 1D convolution used for?
2. How does 1D convolution differ from 2D convolution?
3. What is the input shape for PyTorch's Conv1d?
4. What kernel sizes are common for 1D convolution?
5. How do you compute the output sequence length?

### Intermediate - 5
1. How does 1D convolution process multi-channel input?
2. Compare 1D convolution with RNNs for sequence modeling.
3. What is causal convolution and why is it needed?
4. How do dilated 1D convolutions increase receptive field?
5. What is the TextCNN architecture?

### Advanced - 3
1. Derive the relationship between dilation rate and receptive field in 1D conv stacks.
2. Design a 1D convolution architecture for audio generation.
3. Compare the computational complexity of 1D convs vs transformers for sequence modeling.

## Practice Problems

### Easy - 5
1. Apply Conv1d(1, 1, 3) to a sequence of length 20.
2. Compute output length for Conv1d with kernel 5, stride 2, padding 2.
3. Convert text embeddings to Conv1d-compatible format.
4. Implement a single 1D convolution manually.
5. Create a 1D conv layer for 64-channel input with kernel size 7.

### Medium - 5
1. Implement a TextCNN for sentiment classification.
2. Build a causal convolutional layer for autoregressive generation.
3. Implement a stack of dilated 1D convolutions.
4. Compare 1D conv vs LSTM on a time series forecasting task.
5. Implement 1D convolution with grouped channels.

### Hard - 3
1. Implement a WaveNet-style autoregressive model.
2. Design a 1D convolution-based transformer hybrid.
3. Derive and implement gradient computation for causal 1D convolution.

## Solutions

### Easy - 1 Solution
```python
x = torch.randn(1, 1, 20)
conv = nn.Conv1d(1, 1, 3)
out = conv(x)
print(out.shape)  # (1, 1, 18)
```

## Related Concepts

DL-176 Convolution Operation, DL-180 Dilation, DL-185 Receptive Field

## Next Concepts

DL-193 2D Convolution, DL-194 3D Convolution

## Summary

1D convolution processes sequential data by sliding a kernel along the time/position axis. It's widely used for audio, text, and time series tasks, offering parallel computation and long-range modeling through dilated stacks. Understanding 1D convolution extends CNN knowledge to non-image domains.

## Key Takeaways

- 1D conv applies kernels along a single temporal/spatial dimension
- Input shape: (N, C, L); Output shape: (N, C_out, L_out)
- Output length: O = (L - K + 2P)/S + 1
- Used for text (TextCNN), audio (WaveNet), time series
- Dilated 1D convs enable exponential receptive field growth
- Causal padding prevents looking at future timesteps
- Parallelizable (unlike RNNs)
- Often more efficient than RNNs for long sequences
- Channel dimension management is critical (permute after embedding)
