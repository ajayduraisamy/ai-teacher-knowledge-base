# Concept: Experiment Tracking

## Concept ID

DL-161

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the importance of systematic experiment tracking
- Implement experiment tracking using manual logging and TensorBoard
- Compare different experiment tracking tools (MLflow, Weights and Biases)
- Organize experiments with consistent naming and metadata
- Analyze experiment results to guide model development

## Prerequisites

- Training loop (DL-156)
- Validation loop (DL-157)
- Model saving (DL-160)
- Understanding of hyperparameters

## Definition

Experiment tracking is the systematic recording of all experimental configurations, metrics, model artifacts, and results during the model development process. It enables comparing different runs, reproducing results, identifying the best configurations, and maintaining a history of all experiments. Modern experiment tracking tools (MLflow, Weights and Biases, Neptune) provide dashboards for visualizing and comparing runs, while manual approaches use log files and spreadsheets.

## Intuition

Imagine running a science experiment without a lab notebook. You might forget which temperature you used, which procedure you followed, or what the results were. Experiment tracking is the digital lab notebook for machine learning. It records everything: hyperparameters, code version, dataset version, metrics at each epoch, trained model artifacts, and even the hardware used. When you have run 100+ experiments, having a systematic tracking system is the difference between reproducible science and chaotic guesswork.

## Why This Concept Matters

Without experiment tracking, deep learning development becomes: (1) unreproducible — you cannot recreate your best results, (2) inefficient — you repeat failed experiments, (3) opaque — you cannot explain why certain configurations worked, and (4) disorganized — you waste time searching through notebooks and files. Professional ML teams use experiment tracking as their primary development tool. It is essential for publication reproducibility, team collaboration, and production model governance.

## Code Examples

### Example 1: Manual Experiment Tracking

`python
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime

class ManualExperimentTracker:
    def __init__(self, experiment_name, base_dir='experiments'):
        self.experiment_dir = os.path.join(base_dir, experiment_name, 
                                           datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.metrics = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.config = {}

    def log_config(self, config):
        self.config = config
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def log_metrics(self, epoch, train_loss, val_loss=None, val_acc=None):
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if val_acc is not None:
            self.metrics['val_acc'].append(val_acc)
        
        # Save to file after each epoch
        with open(os.path.join(self.experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_model(self, model, epoch):
        path = os.path.join(self.experiment_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), path)

    def get_summary(self):
        summary = {
            'config': self.config,
            'best_val_loss': min(self.metrics['val_loss'], default=None),
            'best_val_acc': max(self.metrics['val_acc'], default=None),
            'n_epochs': len(self.metrics['train_loss']),
        }
        return summary

tracker = ManualExperimentTracker('lr_test')
tracker.log_config({'lr': 0.01, 'batch_size': 32, 'epochs': 5, 'model': 'SimpleNet'})

for epoch in range(5):
    train_loss = 1.0 / (epoch + 1) + 0.1
    val_loss = 0.8 / (epoch + 1) + 0.01 * epoch
    val_acc = 0.3 + 0.05 * epoch
    tracker.log_metrics(epoch, train_loss, val_loss, val_acc)

summary = tracker.get_summary()
print(f"Experiment summary: {json.dumps(summary, indent=2)}")
# Output:
# Experiment summary: {
#   "config": {"lr": 0.01, "batch_size": 32, "epochs": 5, "model": "SimpleNet"},
#   "best_val_loss": 0.1608,
#   "best_val_acc": 0.5,
#   "n_epochs": 5
# }
`

### Example 2: TensorBoard Integration

`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TensorBoardTracker:
    def __init__(self, log_dir='runs'):
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, run_name))
        print(f"TensorBoard logs in: {os.path.join(log_dir, run_name)}")

    def log_config(self, config):
        config_str = ', '.join(f'{k}={v}' for k, v in config.items())
        self.writer.add_text('config', config_str)
        self.config = config
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def log_epoch(self, model, epoch, train_loss, val_loss=None, val_acc=None):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        
        if val_loss is not None:
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.writer.add_scalar('Best/val_loss', val_loss, epoch)
        
        if val_acc is not None:
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.writer.add_scalar('Best/val_acc', val_acc, epoch)
        
        # Log histogram of weight distributions
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, epoch)

    def log_lr(self, lr, epoch):
        self.writer.add_scalar('LR', lr, epoch)

    def close(self):
        self.writer.close()

tracker = TensorBoardTracker()
tracker.log_config({'lr': 0.01, 'model': 'ResNet18', 'dataset': 'CIFAR-10'})

model = nn.Linear(10, 2)
for epoch in range(5):
    train_loss = 1.0 / (epoch + 1)
    val_loss = 0.8 / (epoch + 1)
    val_acc = 0.5 + 0.1 * epoch
    tracker.log_epoch(model, epoch, train_loss, val_loss, val_acc)
    tracker.log_lr(0.01 * (0.9 ** epoch), epoch)

tracker.close()
print("TensorBoard logging complete. Run: tensorboard --logdir=runs")
# Output:
# TensorBoard logs in: runs/20240101_120000
# TensorBoard logging complete. Run: tensorboard --logdir=runs
`

### Example 3: Experiment Comparison

`python
import json

class ExperimentComparator:
    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        self.results = {}

    def add_experiment(self, name, config, metrics):
        self.results[name] = {
            'config': config,
            'best_val_loss': min(metrics.get('val_loss', [float('inf')])),
            'best_val_acc': max(metrics.get('val_acc', [0.0])),
            'final_train_loss': metrics.get('train_loss', [])[-1] if metrics.get('train_loss') else None,
        }

    def compare(self):
        print(f"{'Experiment':20s} {'LR':8s} {'Batch':8s} {'Best Val Loss':15s} {'Best Acc':10s}")
        print("-" * 65)
        
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['best_val_acc'], reverse=True)
        
        for name, result in sorted_results:
            config = result['config']
            lr = str(config.get('lr', 'N/A'))
            batch = str(config.get('batch_size', 'N/A'))
            loss = f"{result['best_val_loss']:.4f}"
            acc = f"{result['best_val_acc']:.2%}"
            print(f"{name:20s} {lr:8s} {batch:8s} {loss:15s} {acc:10s}")

comparator = ExperimentComparator()

# Simulate multiple experiments
experiments = [
    ('LR_0.01_BS_32', {'lr': 0.01, 'batch_size': 32}, 
     {'train_loss': [1.0, 0.5, 0.3], 'val_loss': [0.8, 0.6, 0.5], 'val_acc': [0.4, 0.5, 0.55]}),
    ('LR_0.001_BS_32', {'lr': 0.001, 'batch_size': 32}, 
     {'train_loss': [1.5, 0.8, 0.6], 'val_loss': [1.2, 0.7, 0.55], 'val_acc': [0.3, 0.45, 0.52]}),
    ('LR_0.01_BS_64', {'lr': 0.01, 'batch_size': 64}, 
     {'train_loss': [1.1, 0.6, 0.4], 'val_loss': [0.9, 0.63, 0.48], 'val_acc': [0.38, 0.48, 0.58]}),
]

for name, config, metrics in experiments:
    comparator.add_experiment(name, config, metrics)

comparator.compare()
# Output:
# Experiment            LR       Batch    Best Val Loss   Best Acc
# -----------------------------------------------------------------
# LR_0.01_BS_64       0.01     64        0.4800          58.00%
# LR_0.01_BS_32       0.01     32        0.5000          55.00%
# LR_0.001_BS_32      0.001    32        0.5500          52.00%
`

## Common Mistakes

1. **Not tracking enough metadata**: Record dataset version, code commit hash, random seed, and hardware details — not just hyperparameters.
2. **Inconsistent naming conventions**: Use structured experiment names (e.g., 'model_dataset_lr_wd_seed') for easy filtering.
3. **Not tracking failed experiments**: Record failures too — they contain valuable information about what does not work.
4. **Overwriting results**: Use timestamps or unique IDs for each run. Never overwrite previous experiment results.
5. **Manual logging errors**: Automate tracking to avoid human errors in recording metrics and configurations.

## Interview Questions

### Beginner

1. What is experiment tracking and why is it important?
2. What information should be tracked for each experiment?
3. Name three experiment tracking tools.
4. How do you avoid overwriting previous experiment results?
5. What is a run in experiment tracking?

### Intermediate

1. Compare TensorBoard with MLflow for experiment tracking.
2. How would you track experiments across a team of researchers?
3. What metadata beyond hyperparameters should be recorded?
4. How do you handle experiment reproducibility across different hardware?
5. Design a naming convention for experiments.

### Advanced

1. Design an automated experiment comparison system that suggests the next configuration to try.
2. Implement a hierarchical experiment tracking system that groups related runs.
3. How would you track and compare experiments with different evaluation metrics?

## Practice Problems

### Easy

1. Set up TensorBoard logging for training loss and validation accuracy.
2. Create a JSON log file for experiment configurations.
3. Implement automatic run naming with timestamps.
4. Log hyperparameters alongside metrics.
5. Create a simple experiment comparison table.

### Medium

1. Integrate MLflow into a training pipeline.
2. Implement experiment tracking with automated git commit logging.
3. Create a visualization dashboard for comparing multiple runs.
4. Implement a search function to find experiments by configuration.
5. Add hardware monitoring (GPU utilization, memory) to experiment tracking.

### Hard

1. Design a distributed experiment tracking system for a cluster of researchers.
2. Implement an automated experiment suggestion system based on Bayesian optimization.
3. Create a full-featured experiment management platform with user authentication and model registry.

## Solutions

### Easy Solutions

1. Use SummaryWriter from torch.utils.tensorboard
2. Save config as JSON and metrics as separate JSON per epoch
3. Use datetime.now().strftime('%Y%m%d_%H%M%S') as run name
4. Add_text or add_hparams in TensorBoard
5. Print formatted table sorted by best metric

## Related Concepts

- Hyperparameter Search (DL-162)
- Model Saving and Loading (DL-160)
- Checkpointing (DL-159)
- Learning Curves (DL-166)

## Next Concepts

- Hyperparameter Search (DL-162)
- Grid Search (DL-163)
- Random Search (DL-164)

## Summary

Experiment tracking systematically records configurations, metrics, and artifacts for every training run. It enables reproducible research, efficient comparison, and informed decision-making. Tools range from manual JSON logging to sophisticated platforms like MLflow and Weights and Biases.

## Key Takeaways

- Track everything: config, metrics, code version, dataset, hardware
- Use consistent naming conventions for experiments
- Automate logging to avoid human errors
- Record failed experiments too
- TensorBoard for local visualization, MLflow/W&B for team collaboration
- Store model artifacts alongside metrics
- Compare experiments to guide development decisions
- Reproducibility depends on thorough experiment tracking
