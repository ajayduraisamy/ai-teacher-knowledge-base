# Concept: Model Saving and Loading

## Concept ID

DL-160

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand different PyTorch saving formats (state_dict, full model, TorchScript)
- Implement model saving and loading for inference and training
- Handle device mapping when loading models (CPU, CUDA, multi-GPU)
- Manage model versioning and compatibility
- Implement best practices for model serialization

## Prerequisites

- Checkpointing (DL-159)
- PyTorch model definition
- Understanding of inference vs training
- Device management (CPU/GPU)

## Definition

Model saving and loading refers to the process of persisting trained neural network weights to disk and restoring them for inference or further training. PyTorch offers multiple approaches: saving only the state_dict (recommended), saving the full model object (convenient but brittle), and exporting to TorchScript or ONNX for production deployment. Proper saving and loading practices ensure model portability, reproducibility, and production readiness.

## Intuition

Think of model saving like saving a recipe versus saving a cooked meal. The state_dict is the recipe — the ingredients (weights) and quantities (layer dimensions). It is lightweight, portable, and can be used to recreate the dish in any kitchen (any compatible architecture). Saving the full model is like freezing the entire meal — convenient if you are consuming it immediately, but fragile if the freezer (PyTorch version) changes. For production, you want a standardized format (TorchScript/ONNX) like a TV dinner — it works the same way everywhere.

## Why This Concept Matters

Proper model saving and loading is essential for: (1) deploying models to production, (2) sharing models with collaborators, (3) restoring models for continued training, (4) model versioning and reproducibility, (5) comparing different trained models, and (6) distributing pre-trained models (Hugging Face, PyTorch Hub). Mistakes in saving/loading can lead to incorrect predictions, training failures, or wasted time debugging device mismatches.

## Code Examples

### Example 1: Basic state_dict Save and Load

`python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Save
model = SimpleModel()
torch.save(model.state_dict(), 'model_state.pt')
print(f"Saved state_dict to 'model_state.pt'")
print(f"state_dict keys: {model.state_dict().keys()}")

# Load (need to instantiate model first)
loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load('model_state.pt'))
loaded_model.eval()
print(f"Loaded model parameters match: "
      f"{all((p1 == p2).all() for p1, p2 in zip(model.parameters(), loaded_model.parameters()))}")
# Output:
# Saved state_dict to 'model_state.pt'
# state_dict keys: odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
# Loaded model parameters match: True
`

### Example 2: Full Model Save (Not Recommended)

`python
import torch
import torch.nn as nn

model = SimpleModel()

# Full model save (includes architecture definition)
torch.save(model, 'full_model.pt')
print("Saved full model (not recommended)")

# Loading full model (class must be importable)
loaded_full = torch.load('full_model.pt')
loaded_full.eval()
print(f"Full model loaded successfully")

# Problem: if SimpleModel class is not defined in this scope
try:
    # This would fail if SimpleModel is not defined
    pass
except Exception as e:
    print(f"Error: {e}")

# Using strict=False for partial loading
partial_state = {k: v for k, v in model.state_dict().items() if 'fc1' in k}
model2 = SimpleModel()
model2.load_state_dict(partial_state, strict=False)
print(f"Loaded partial state dict (fc1 only)")
# Output:
# Saved full model (not recommended)
# Full model loaded successfully
# Loaded partial state dict (fc1 only)
`

### Example 3: Device Management

`python
import torch
import torch.nn as nn

model = SimpleModel()

# 1. Save and load to same device
torch.save(model.state_dict(), 'model_cpu.pt')
state = torch.load('model_cpu.pt')
model_cpu = SimpleModel()
model_cpu.load_state_dict(state)
print(f"Loaded to CPU: {next(model_cpu.parameters()).device}")

# 2. Save on GPU, load on CPU
if torch.cuda.is_available():
    model_gpu = SimpleModel().cuda()
    torch.save(model_gpu.state_dict(), 'model_gpu.pt')
    state = torch.load('model_gpu.pt', map_location='cpu')
    model_from_gpu = SimpleModel()
    model_from_gpu.load_state_dict(state)
    print(f"Loaded GPU-trained model to CPU: success")

# 3. DataParallel model saving
dp_model = nn.DataParallel(SimpleModel())
dp_state = dp_model.state_dict()  # module. prefix
# Remove 'module.' prefix if needed when loading to non-DP model
clean_state = {k.replace('module.', ''): v for k, v in dp_state.items()}
model_clean = SimpleModel()
model_clean.load_state_dict(clean_state, strict=False)
print(f"DataParallel state loaded to single GPU model: "
      f"{len(clean_state)} keys matched")

# 4. map_location examples
state = torch.load('model_cpu.pt')
# Automatically handle device mapping
model_auto = SimpleModel()
model_auto.load_state_dict(state)
# Output:
# Loaded to CPU: cpu
# DataParallel state loaded to single GPU model: 4 keys matched
`

## Common Mistakes

1. **Saving the full model instead of state_dict**: Full model saves are tied to the exact class definition and PyTorch version, causing compatibility issues.
2. **Forgetting model.eval() after loading**: Models loaded for inference should be set to eval mode to disable dropout and freeze batch norm.
3. **Not handling DataParallel wrapper**: DataParallel prepends 'module.' to parameter names. Strip this prefix when loading to a non-wrapped model.
4. **Device mismatch errors**: Loading a GPU-trained model on CPU requires map_location='cpu'. Device names may differ across machines.
5. **Using torch.load with weights_only=False on untrusted sources**: Loading pickled files from untrusted sources can execute arbitrary code.

## Interview Questions

### Beginner

1. What is a state_dict in PyTorch?
2. What is the recommended way to save models in PyTorch?
3. How do you load a saved model for inference?
4. What is the purpose of model.eval() after loading?
5. What is the difference between saving state_dict and full model?

### Intermediate

1. How do you handle device mapping when loading a model trained on GPU to CPU?
2. How does DataParallel affect model saving and loading?
3. What is the strict parameter in load_state_dict and when would you set it to False?
4. How do you save and load optimizer state for resuming training?
5. What is TorchScript and when should you use it?

### Advanced

1. Design a model versioning system that handles backward compatibility.
2. Implement a custom serialization format for models larger than available RAM.
3. How would you handle cross-version PyTorch compatibility for saved models?

## Practice Problems

### Easy

1. Save a model's state_dict to a file.
2. Load a state_dict and restore the model.
3. Save a checkpoint with optimizer state.
4. Load a model and set it to evaluation mode.
5. Save a model using map_location='cpu' for portability.

### Medium

1. Implement a function that saves and loads models robustly (handles device, DataParallel, strict).
2. Convert a saved PyTorch model to TorchScript.
3. Implement model versioning with metadata (architecture, date, metrics).
4. Load a pre-trained model from a URL.
5. Fix a state_dict key mismatch between DataParallel and non-DataParallel models.

### Hard

1. Implement a model serialization format that preserves the computation graph for visualization.
2. Design a distributed model loading system for large-scale inference.
3. Implement a model registry with automatic versioning and rollback capabilities.

## Solutions

### Easy Solutions

1. torch.save(model.state_dict(), 'model.pt')
2. model = MyModel(); model.load_state_dict(torch.load('model.pt')); model.eval()
3. torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, 'checkpoint.pt')
4. model.eval()
5. torch.save(model.cpu().state_dict(), 'model_cpu.pt')

## Related Concepts

- Checkpointing (DL-159)
- Training Loop (DL-156)
- TorchScript
- ONNX Export

## Next Concepts

- Experiment Tracking (DL-161)
- Hyperparameter Search (DL-162)
- Grid Search (DL-163)

## Summary

Model saving and loading in PyTorch is best done using state_dict files, which are portable and version-independent. Key considerations include device mapping, DataParallel handling, and setting the correct model mode after loading.

## Key Takeaways

- Always save state_dict, not the full model object
- Use model.eval() after loading for inference
- Handle device mapping with map_location parameter
- Strip 'module.' prefix from DataParallel state dicts
- Save optimizer state for training resume
- Use strict=False for loading partial checkpoints
- Export to TorchScript/ONNX for production deployment
- Save metadata (architecture, version, metrics) alongside weights
