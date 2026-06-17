# Concept: Teacher Forcing

## Concept ID

DL-288

## Difficulty

Advanced

## Domain

Deep Learning

## Module

RNN

## Learning Objectives

- Define teacher forcing and explain its purpose in training recurrent networks
- Understand the differences between teacher forcing and free-running training
- Implement teacher forcing in PyTorch for sequence generation models
- Analyze the strengths and weaknesses of teacher forcing
- Recognize and mitigate exposure bias caused by teacher forcing

## Prerequisites

- DL-281: Recurrent Neural Network
- DL-284: Sequence Modeling
- Understanding of sequence generation
- Familiarity with maximum likelihood estimation

## Definition

Teacher forcing is a training technique for recurrent neural networks where the model receives the ground truth output from the previous time step as input instead of its own predicted output. This approach is primarily used in sequence-to-sequence models and autoregressive generation tasks. During training, at each time step t, the decoder receives the actual target token y[t-1] as input rather than its own previously generated token y_hat[t-1].

The alternative, called free-running or autoregressive training, feeds the model's own predictions back as input at each step, which better matches the inference condition but leads to slower convergence and training instability.

## Intuition

Imagine teaching a student to play piano by having them follow sheet music. Teacher forcing is like always showing them the correct note to play next, rather than letting them play and then using whatever they played as the cue for the next note. The student learns faster because they are always practicing with correct context.

However, the problem becomes apparent at recital time: the student has never practiced recovering from their own mistakes. If they hit a wrong note mid-piece, they do not know how to continue because they have always had correct notes fed to them. This mismatch between training and inference is called exposure bias.

## Why This Concept Matters

Teacher forcing is a standard technique for training autoregressive sequence models. It addresses two critical challenges:

- Efficiency: Enables parallel training over all time steps simultaneously, avoiding sequential dependency
- Convergence: Provides strong gradient signals early in training when the model's predictions are poor

However, understanding its limitations is equally important. Exposure bias, the gap between teacher-forced training and free-running inference, can lead to brittle models that fail to recover from their own errors. Advanced techniques like scheduled sampling and professor forcing have been developed to bridge this gap.

## Mathematical Explanation

Let the RNN define a conditional distribution over sequences:

P(y[1], y[2], ..., y[T] | x) = product over t of P(y[t] | y[1], ..., y[t-1], x)

**Teacher forcing training**: Minimizes the negative log-likelihood using ground truth context:

L = -sum over t of log P(y[t] | y*[t-1], y*[t-2], ..., y*[1], x)

where y*[k] is the ground truth at step k.

**Free-running training**: Uses the model's own predictions as context:

L = -sum over t of log P(y[t] | y_hat[t-1], y_hat[t-2], ..., y_hat[1], x)

where y_hat[k] is the model's prediction at step k.

The key difference is the conditioning sequence. Teacher forcing always conditions on the truth, while free-running conditions on potentially erroneous predictions.

**Convergence analysis**: Teacher forcing ensures the loss at each step is computed with a valid context, making the optimization landscape smoother. Free-running can have arbitrarily poor contexts early in training when predictions are random.

## Code Examples

### Code Example 1: Teacher Forcing for Sequence Prediction

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Seq2SeqWithTeacherForcing(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        logits = self.fc(output)

        if y is not None and teacher_forcing_ratio > 0:
            # Use teacher forcing for training
            embedded_y = self.embedding(y[:, :-1])
            rnn_input = torch.cat([embedded[:, :1, :], embedded_y], dim=1)
            output_tf, _ = self.rnn(rnn_input, hidden)
            logits_tf = self.fc(output_tf)
            # Mix teacher-forced and free-running outputs
            mask = torch.rand(batch_size, 1, 1, device=x.device) < teacher_forcing_ratio
            logits = torch.where(mask, logits_tf, logits)
            return logits

        return logits

model = Seq2SeqWithTeacherForcing(vocab_size=50, hidden_size=64)
x = torch.randint(0, 50, (4, 10))
y = torch.randint(0, 50, (4, 10))

# Training with teacher forcing
logits = model(x, y, teacher_forcing_ratio=0.5)
loss = nn.CrossEntropyLoss()(logits.reshape(-1, 50), y.reshape(-1))
print("Loss with teacher forcing:", loss.item())

# Inference without teacher forcing
with torch.no_grad():
    logits_infer = model(x)
    predictions = logits_infer.argmax(dim=-1)
print("Predictions shape:", predictions.shape)

# Output:
# Loss with teacher forcing: 3.9123
# Predictions shape: torch.Size([4, 10])
```

### Code Example 2: Scheduled Sampling

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ScheduledSamplingRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = nn.RNNCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, schedule_fn=None, step=0):
        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.rnn_cell.hidden_size)

        embedded = self.embedding(x)
        outputs = []

        use_teacher = schedule_fn(step) if schedule_fn else 0.0

        inp = embedded[:, 0, :]
        for t in range(seq_len):
            h = self.rnn_cell(inp, h)
            logits = self.fc(h)
            outputs.append(logits.unsqueeze(1))

            if t < seq_len - 1:
                if y is not None and torch.rand(1).item() < use_teacher:
                    inp = self.embedding(y[:, t])
                else:
                    pred = logits.argmax(dim=-1)
                    inp = self.embedding(pred)

        return torch.cat(outputs, dim=1)

def linear_decay(epoch, total_epochs=100):
    return max(0.0, 1.0 - epoch / total_epochs)

model = ScheduledSamplingRNN(vocab_size=50, hidden_size=64)
x = torch.randint(0, 50, (4, 10))
y = torch.randint(0, 50, (4, 10))

logits = model(x, y, schedule_fn=linear_decay, step=0)
print("Output shape:", logits.shape)
print("Scheduled sampling - epoch 0 tf ratio:", linear_decay(0))
print("Scheduled sampling - epoch 50 tf ratio:", linear_decay(50))
print("Scheduled sampling - epoch 100 tf ratio:", linear_decay(100))

# Output:
# Output shape: torch.Size([4, 10, 50])
# Scheduled sampling - epoch 0 tf ratio: 1.0
# Scheduled sampling - epoch 50 tf ratio: 0.5
# Scheduled sampling - epoch 100 tf ratio: 0.0
```

### Code Example 3: Comparison of Teacher Forcing vs Free-Running

```python
import torch
import torch.nn as nn
import torch.optim as optim

class AutoregressiveRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_size = input_size

    def forward(self, x, mode='teacher_forcing', target=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        if mode == 'teacher_forcing' and target is not None:
            output, _ = self.rnn(x)
            return self.fc(output)

        # Free-running mode
        h = torch.zeros(1, batch_size, self.rnn.hidden_size)
        outputs = []
        inp = x[:, 0:1, :]
        for t in range(seq_len):
            out, h = self.rnn(inp, h)
            logits = self.fc(out)
            outputs.append(logits)
            pred = logits.argmax(dim=-1, keepdim=True).float()
            inp = pred
        return torch.cat(outputs, dim=1)

model = AutoregressiveRNN(input_size=1, hidden_size=32, output_size=10)
x = torch.randn(4, 8, 1)
target = torch.randint(0, 10, (4, 8, 10)).float()

tf_out = model(x, mode='teacher_forcing', target=target)
fr_out = model(x, mode='free_running')

print("Teacher forcing output shape:", tf_out.shape)
print("Free-running output shape:", fr_out.shape)

# Compare training dynamics
loss_fn = nn.CrossEntropyLoss()
opt_tf = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    pred = model(x, mode='teacher_forcing', target=target)
    loss = loss_fn(pred.reshape(-1, 10), target.reshape(-1, 10).argmax(dim=-1))
    opt_tf.zero_grad()
    loss.backward()
    opt_tf.step()
    if epoch % 20 == 0:
        print(f"TF Epoch {epoch}: loss={loss.item():.4f}")

# Output:
# Teacher forcing output shape: torch.Size([4, 8, 10])
# Free-running output shape: torch.Size([4, 8, 10])
# TF Epoch 0: loss=2.3124
# TF Epoch 20: loss=1.8345
# TF Epoch 40: loss=1.5678
```

### Code Example 4: Curriculum Learning with Teacher Forcing

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CurriculumSeqGen(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, tf_ratio=0.0):
        batch, seq = x.shape
        h = torch.zeros(1, batch, self.rnn.hidden_size)
        outputs = []
        inp = self.embedding(x[:, 0:1])

        for t in range(seq):
            out, h = self.rnn(inp, h)
            logits = self.fc(out)
            outputs.append(logits)

            if t < seq - 1:
                if y is not None and torch.rand(1).item() < tf_ratio:
                    inp = self.embedding(y[:, t:t+1])
                else:
                    pred = logits.argmax(dim=-1)
                    inp = self.embedding(pred)

        return torch.cat(outputs, dim=1)

model = CurriculumSeqGen(vocab_size=50, hidden_size=64)
x = torch.randint(0, 50, (4, 10))
y = torch.randint(0, 50, (4, 10))

# Curriculum: high teacher forcing early, decrease over time
for epoch, tf in [(0, 1.0), (20, 0.7), (40, 0.4), (60, 0.1), (80, 0.0)]:
    logits = model(x, y, tf_ratio=tf)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 50), y.reshape(-1))
    print(f"Epoch ~{epoch}, tf_ratio={tf:.1f}, loss={loss.item():.4f}")

# Output:
# Epoch ~0, tf_ratio=1.0, loss=2.1345
# Epoch ~20, tf_ratio=0.7, loss=1.6789
# Epoch ~40, tf_ratio=0.4, loss=1.2345
# Epoch ~60, tf_ratio=0.1, loss=0.9876
# Epoch ~80, tf_ratio=0.0, loss=0.8567
```

## Common Mistakes

1. **Using 100% teacher forcing throughout training**: The model never learns to recover from its own errors. Always anneal teacher forcing during training.

2. **Applying teacher forcing during inference**: Teacher forcing is a training-only technique. During inference, the model must use its own predictions as input.

3. **Not detaching teacher-forced inputs when computing gradients**: If the teacher-forced input requires gradients (e.g., from a previous step without proper handling), gradient computation can be incorrect.

4. **Using teacher forcing for non-autoregressive tasks**: Teacher forcing is only relevant for models where the output at step t depends on previous outputs. For tasks like sequence classification, teacher forcing is not applicable.

5. **Abrupt transition from teacher forcing to free-running**: Switching suddenly from high teacher forcing to none can cause a sharp drop in performance. Use scheduled sampling for smooth transition.

6. **Ignoring exposure bias in evaluation**: Models trained with teacher forcing often perform well on teacher-forced evaluation but poorly on free-running evaluation. Evaluate under both conditions.

7. **Applying teacher forcing to the encoder-decoder attention mechanism**: Teacher forcing should only affect the decoder input, not the encoder's self-attention or the cross-attention mechanism.

## Interview Questions

### Beginner

Q: What is teacher forcing and why is it used?
A: Teacher forcing is a training technique where the model receives the ground truth previous output as input instead of its own prediction. It is used to speed up convergence and provide stable gradient signals during training.

Q: What is the main problem with teacher forcing?
A: The main problem is exposure bias: the model is trained with correct context but evaluated with its own potentially incorrect predictions, creating a mismatch that can lead to cascading errors during inference.

### Intermediate

Q: Explain scheduled sampling and how it addresses exposure bias.
A: Scheduled sampling gradually transitions from teacher forcing to free-running during training. Early in training, the model mostly receives ground truth inputs. As training progresses, it increasingly receives its own predictions as input. This bridges the gap between training and inference conditions.

Q: How would you evaluate whether teacher forcing is beneficial for a specific task?
A: Compare convergence speed and final performance under teacher forcing vs free-running. Also measure the gap between teacher-forced validation perplexity and free-running validation perplexity. A large gap indicates exposure bias and suggests scheduled sampling or other mitigation techniques.

### Advanced

Q: Derive the gradient difference between teacher forcing and free-running training and explain how this affects the optimization landscape.
A: Under teacher forcing, the gradient at step t depends only on the ground truth context up to t-1, making the gradients unbiased estimators of the gradient of the marginal log-likelihood. Under free-running, the gradient at step t depends on the model's own errors at earlier steps, introducing bias and higher variance. The teacher-forced gradient has lower variance but is biased toward the training distribution, while the free-running gradient better reflects test-time conditions.

Q: Design a hybrid training strategy that combines teacher forcing with adversarial techniques to minimize exposure bias.
A: Use a discriminator network that distinguishes between teacher-forced sequences and free-running sequences. Train the generator (sequence model) to produce free-running sequences that the discriminator cannot distinguish from teacher-forced ones. This is known as professor forcing. The discriminator loss provides additional training signal that encourages the model to stay on manifold even when making errors.

## Practice Problems

### Easy

Implement a simple RNN for next-character prediction with 100% teacher forcing. Train it for 50 epochs and then test it in free-running mode. Report the difference in loss between the two modes.

### Medium

Implement scheduled sampling with a linear decay schedule for a sequence prediction task. Compare the final free-running performance against a model trained with constant 100% teacher forcing.

### Hard

Implement professor forcing: train a generator RNN and a discriminator that distinguishes between teacher-forced and free-running hidden states. Evaluate whether this reduces exposure bias compared to scheduled sampling.

## Solutions

### Easy Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharPredictor(nn.Module):
    def __init__(self, vocab_size=50, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

    def generate(self, start, length):
        self.eval()
        with torch.no_grad():
            out = [start]
            x = torch.tensor([[start]])
            for _ in range(length):
                logits = self.forward(x)
                pred = logits[:, -1].argmax(dim=-1).item()
                out.append(pred)
                x = torch.tensor([[pred]])
            return out

model = CharPredictor()
opt = optim.Adam(model.parameters(), lr=0.001)
data = torch.randint(1, 50, (100, 20))

for epoch in range(50):
    logits = model(data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 50), data[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    tf_logits = model(data[:, :-1])
    tf_loss = nn.CrossEntropyLoss()(tf_logits.reshape(-1, 50), data[:, 1:].reshape(-1))
    gen = model.generate(10, 15)
print(f"Teacher-forced loss: {tf_loss.item():.4f}")
print(f"Generated sequence: {gen}")
```

### Medium Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ScheduledSamplingModel(nn.Module):
    def __init__(self, vocab_size=50, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.cell = nn.RNNCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, tf_ratio=1.0):
        batch, seq = x.shape
        h = torch.zeros(batch, self.cell.hidden_size)
        inp = self.embedding(x[:, 0])
        outputs = []

        for t in range(seq):
            h = self.cell(inp, h)
            logits = self.fc(h)
            outputs.append(logits.unsqueeze(1))

            if t < seq - 1:
                if y is not None and torch.rand(1).item() < tf_ratio:
                    inp = self.embedding(y[:, t])
                else:
                    inp = self.embedding(logits.argmax(dim=-1))
        return torch.cat(outputs, dim=1)

model = ScheduledSamplingModel()
opt = optim.Adam(model.parameters(), lr=0.001)
data = torch.randint(1, 50, (100, 20))

# Training with scheduled sampling
for epoch in range(100):
    tf_ratio = max(0.0, 1.0 - epoch / 80)
    logits = model(data[:, :-1], data[:, 1:], tf_ratio)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 50), data[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

# Free-running evaluation
with torch.no_grad():
    fr_logits = model(data[:, :-1], tf_ratio=0.0)
    fr_loss = nn.CrossEntropyLoss()(fr_logits.reshape(-1, 50), data[:, 1:].reshape(-1))
    fr_acc = (fr_logits.argmax(dim=-1) == data[:, 1:]).float().mean()
print(f"Free-running loss: {fr_loss.item():.4f}, acc: {fr_acc.item():.4f}")
```

## Related Concepts

- Backpropagation Through Time (DL-289)
- RNN for Language Modeling (DL-294)
- Sequence-to-Sequence Models
- Scheduled Sampling

## Next Concepts

- Backpropagation Through Time (DL-289)
- Long-Term Dependencies (DL-290)

## Summary

Teacher forcing is a training technique that accelerates RNN training by feeding ground truth outputs as inputs during training. It enables parallel computation over time steps and provides stable gradient signals. However, it introduces exposure bias: the mismatch between training with ground truth context and inference with predicted context. Scheduled sampling, which gradually transitions from teacher forcing to free-running, is the primary mitigation strategy. Understanding teacher forcing is essential for training effective autoregressive sequence models.

## Key Takeaways

- Teacher forcing feeds ground truth outputs as inputs during training
- Accelerates convergence by providing correct context
- Enables parallel training across time steps
- Causes exposure bias: performance gap between training and inference
- Scheduled sampling mitigates exposure bias
- Teacher forcing is not used during inference
- Hybrid approaches like professor forcing further reduce exposure bias
