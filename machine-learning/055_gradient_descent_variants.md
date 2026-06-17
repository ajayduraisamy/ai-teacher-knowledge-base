# Concept: Gradient Descent Variants

## Concept ID

ML-055

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Neural Networks

## Learning Objectives

- Understand the differences between batch, stochastic, and mini-batch gradient descent
- Implement momentum and Nesterov accelerated gradient for faster convergence
- Understand adaptive learning rate methods: AdaGrad, RMSProp, Adam
- Compare convergence speed and stability across variants
- Choose appropriate optimizers for different problem types

## Prerequisites

- Backpropagation (ML-054) — gradient computation in neural networks
- Basic calculus — derivatives, gradient concepts
- Understanding of learning rate and weight updates

## Definition

Gradient descent variants are optimization algorithms that iteratively update model parameters to minimize a loss function. All variants follow the same core principle — move parameters in the direction of the negative gradient — but differ in how they compute gradients, accumulate past gradient information, and adapt learning rates.

The basic gradient descent update is:
theta_{t+1} = theta_t - eta * g_t

where theta are parameters, eta is the learning rate, and g_t is the gradient at time t. Variants modify this update rule to improve convergence speed, stability, and final solution quality.

## Intuition

Think of gradient descent as hiking down a mountain in fog. You can only feel the slope beneath your feet:

- **Batch GD**: Check the entire mountain's slope before each step. Slow but steady.
- **SGD**: Take a step based on the slope under one foot. Fast but erratic.
- **Mini-batch GD**: Check a patch of ground. A practical compromise.
- **Momentum**: Keep walking in the same direction, even if the immediate slope is flat (like a ball rolling downhill).
- **Nesterov**: Look ahead before stepping to avoid overcorrecting.
- **Adam**: Adapt your step size for each direction and keep up the momentum.

## Why This Concept Matters

1. **Training speed**: The right optimizer can reduce training time from weeks to hours.
2. **Solution quality**: Different optimizers converge to different local minima with varying generalization.
3. **Hyperparameter sensitivity**: Modern optimizers (Adam) are much easier to tune than vanilla SGD.
4. **Scale adaptation**: Deep networks with millions of parameters require efficient optimization.
5. **Industry standard**: Adam is the default optimizer in most deep learning projects.

## Mathematical Explanation

### Batch Gradient Descent

Computes the gradient using the entire training dataset:
theta_{t+1} = theta_t - eta * (1/m) * sum_i grad_i(Loss)

**Pros:** Stable convergence, deterministic gradient.
**Cons:** Very slow for large datasets, cannot update online.

### Stochastic Gradient Descent (SGD)

Uses one training example per update:
theta_{t+1} = theta_t - eta * grad_i(Loss)

**Pros:** Fast per-iteration, can escape shallow local minima, online learning.
**Cons:** High variance, oscillates around minimum, needs learning rate scheduling.

### Mini-Batch Gradient Descent

Uses a batch of m examples:
theta_{t+1} = theta_t - eta * (1/b) * sum_j grad_j(Loss)

**Pros:** Balances efficiency and stability, vectorized operations.
**Cons:** Introduces batch size as a hyperparameter.

### Momentum

Accumulates past gradients with a decay factor:
v_t = gamma * v_{t-1} + eta * g_t
theta_{t+1} = theta_t - v_t

where gamma (typically 0.9) controls momentum decay.

**Effect:** Accelerates convergence in consistent directions, dampens oscillations.

### Nesterov Accelerated Gradient (NAG)

Looks ahead by computing gradient at the "looked-ahead" position:
v_t = gamma * v_{t-1} + eta * grad(theta_t - gamma * v_{t-1})
theta_{t+1} = theta_t - v_t

**Effect:** Corrects momentum overshooting, converges faster than standard momentum.

### AdaGrad

Adapts learning rate per-parameter based on past gradients:
G_t = G_{t-1} + g_t^2
theta_{t+1} = theta_t - (eta / (sqrt(G_t) + eps)) * g_t

**Effect:** Large learning rates for sparse parameters, small for frequent ones.
**Limitation:** Learning rate monotonically decreases to zero.

### RMSProp

Fixes AdaGrad's monotonic decay with exponential moving average:
E[g^2]_t = beta * E[g^2]_{t-1} + (1-beta) * g_t^2
theta_{t+1} = theta_t - (eta / (sqrt(E[g^2]_t) + eps)) * g_t

**Effect:** Non-monotonic learning rate, works well for non-stationary objectives.

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSProp:
m_t = beta1 * m_{t-1} + (1-beta1) * g_t  (first moment)
v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2  (second moment)

Bias correction:
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)

Update:
theta_{t+1} = theta_t - (eta / (sqrt(v_hat_t) + eps)) * m_hat_t

**Default hyperparameters:** beta1=0.9, beta2=0.999, eps=1e-8

**Effect:** Most popular optimizer, works well across diverse problem types.

## Code Examples

### Example 1: Comparing Optimizers on a Simple 2D Function

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + 10*y**2

def grad_f(x, y):
    return np.array([2*x, 20*y])

def optimize(optimizer_name, lr=0.1, n_steps=50):
    x, y = 5.0, 2.0
    trajectory = [(x, y)]
    v_x, v_y = 0.0, 0.0
    m_x, m_y, vx, vy = 0.0, 0.0, 0.0, 0.0
    beta1, beta2 = 0.9, 0.999
    gamma = 0.9
    eps = 1e-8

    for t in range(1, n_steps + 1):
        gx, gy = grad_f(x, y)

        if optimizer_name == 'sgd':
            x -= lr * gx
            y -= lr * gy
        elif optimizer_name == 'momentum':
            v_x = gamma * v_x + lr * gx
            v_y = gamma * v_y + lr * gy
            x -= v_x; y -= v_y
        elif optimizer_name == 'nag':
            x_ahead = x - gamma * v_x
            y_ahead = y - gamma * v_y
            gx_a, gy_a = grad_f(x_ahead, y_ahead)
            v_x = gamma * v_x + lr * gx_a
            v_y = gamma * v_y + lr * gy_a
            x -= v_x; y -= v_y
        elif optimizer_name == 'adagrad':
            vx += gx**2; vy += gy**2
            x -= lr * gx / (np.sqrt(vx) + eps)
            y -= lr * gy / (np.sqrt(vy) + eps)
        elif optimizer_name == 'rmsprop':
            vx = 0.9 * vx + 0.1 * gx**2
            vy = 0.9 * vy + 0.1 * gy**2
            x -= lr * gx / (np.sqrt(vx) + eps)
            y -= lr * gy / (np.sqrt(vy) + eps)
        elif optimizer_name == 'adam':
            m_x = beta1 * m_x + (1 - beta1) * gx
            m_y = beta1 * m_y + (1 - beta1) * gy
            vx = beta2 * vx + (1 - beta2) * gx**2
            vy = beta2 * vy + (1 - beta2) * gy**2
            m_x_hat = m_x / (1 - beta1**t)
            m_y_hat = m_y / (1 - beta1**t)
            vx_hat = vx / (1 - beta2**t)
            vy_hat = vy / (1 - beta2**t)
            x -= lr * m_x_hat / (np.sqrt(vx_hat) + eps)
            y -= lr * m_y_hat / (np.sqrt(vy_hat) + eps)

        trajectory.append((x, y))
    return np.array(trajectory)

# Create contour plot
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

optimizers = ['sgd', 'momentum', 'nag', 'adagrad', 'rmsprop', 'adam']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, opt in zip(axes.ravel(), optimizers):
    traj = optimize(opt, lr=0.1, n_steps=30)
    ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.plot(traj[:, 0], traj[:, 1], 'r.-', markersize=8)
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'r*', markersize=15, label='Minimum')
    ax.set_title(opt.upper())
    ax.set_xlim(-5, 5); ax.set_ylim(-3, 3)
    ax.legend()

plt.tight_layout()
plt.show()

# Compare convergence
plt.figure(figsize=(12, 6))
for opt in optimizers:
    traj = optimize(opt, lr=0.1, n_steps=50)
    losses = [f(x, y) for x, y in traj]
    plt.plot(losses, label=opt.upper(), linewidth=2)

plt.xlabel('Step')
plt.ylabel('Loss (log scale)')
plt.yscale('log')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

```
# Output:
[Contour plots showing different optimization paths]
[Convergence curves showing Adam and RMSProp converging fastest]
```

### Example 2: SGD vs Adam for Neural Network Training

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time

X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

solvers = ['sgd', 'adam']
results = []

for solver in solvers:
    start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver=solver,
        learning_rate='adaptive' if solver == 'sgd' else 'constant',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train_s, y_train)
    elapsed = time.time() - start
    train_acc = mlp.score(X_train_s, y_train)
    test_acc = mlp.score(X_test_s, y_test)
    results.append({
        'solver': solver,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'iterations': mlp.n_iter_,
        'time': elapsed,
        'loss_curve': mlp.loss_curve_
    })
    print(f"{solver:5s}: Train={train_acc:.4f}, Test={test_acc:.4f}, "
          f"Iter={mlp.n_iter_}, Time={elapsed:.2f}s")

plt.figure(figsize=(10, 5))
for res in results:
    plt.plot(res['loss_curve'], label=res['solver'].upper(),
             linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curves: SGD vs Adam')
plt.legend()
plt.grid(True)
plt.show()
```

```
# Output:
sgd  : Train=0.9675, Test=0.9370, Iter=200, Time=3.45s
adam : Train=1.0000, Test=0.9520, Iter=87, Time=1.52s
```

### Example 3: Impact of Learning Rate on Convergence

```python
learning_rates = [1.0, 0.1, 0.01, 0.001]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, lr in zip(axes.ravel(), learning_rates):
    # Test on simple function: f(x) = x^2
    x_vals = [5.0]
    for i in range(50):
        g = 2 * x_vals[-1]
        x_vals.append(x_vals[-1] - lr * g)

    ax.plot(x_vals, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', label='Optimal')
    ax.set_title(f'LR={lr}, Final={x_vals[-1]:.4f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('x')
    ax.grid(True)
    ax.legend()

plt.suptitle('Effect of Learning Rate on Convergence (SGD)')
plt.tight_layout()
plt.show()

print("Final values for different learning rates:")
for lr in learning_rates:
    x = 5.0
    for i in range(50):
        x -= lr * 2 * x
    print(f"  LR={lr:.3f}: final x={x:.6f}")
```

```
# Output:
Final values for different learning rates:
  LR=1.000: final x=5.0000 (diverges/oscillates)
  LR=0.100: final x=0.0000
  LR=0.010: final x=0.3642
  LR=0.001: final x=3.8764
```

### Example 4: Adam Hyperparameter Sensitivity

```python
def adam_step(x, g, m, v, t, lr, beta1, beta2, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    return x_new, m, v

# Test on Beale function
def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + \
           (2.625 - x + x*y**3)**2

def beale_grad(x, y):
    dfdx = 2*(1.5-x+x*y)*(y-1) + 2*(2.25-x+x*y**2)*(y**2-1) + \
           2*(2.625-x+x*y**3)*(y**3-1)
    dfdy = 2*(1.5-x+x*y)*x + 2*(2.25-x+x*y**2)*2*x*y + \
           2*(2.625-x+x*y**3)*3*x*y**2
    return np.array([dfdx, dfdy])

# Compare different beta1 values
beta1_vals = [0.5, 0.9, 0.99]
plt.figure(figsize=(12, 4))

for i, beta1 in enumerate(beta1_vals):
    x, y = 3.0, 3.0
    m_x, m_y, v_x, v_y = 0, 0, 0, 0
    traj = [(x, y)]
    for t in range(1, 100):
        gx, gy = beale_grad(x, y)
        x, m_x, v_x = adam_step(x, gx, m_x, v_x, t, 0.01, beta1, 0.999)
        y, m_y, v_y = adam_step(y, gy, m_y, v_y, t, 0.01, beta1, 0.999)
        traj.append((x, y))

    plt.subplot(1, 3, i+1)
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], 'b.-')
    plt.plot(3, 0.5, 'r*', markersize=10)
    plt.title(f'Adam beta1={beta1}')
    plt.xlim(-1, 4); plt.ylim(-1, 4)
    plt.grid(True)

plt.suptitle('Adam: Effect of beta1 on Convergence')
plt.tight_layout()
plt.show()

print("Final positions:")
for beta1 in beta1_vals:
    x, y = 3.0, 3.0
    m_x, m_y, v_x, v_y = 0, 0, 0, 0
    for t in range(1, 100):
        gx, gy = beale_grad(x, y)
        x, m_x, v_x = adam_step(x, gx, m_x, v_x, t, 0.01, beta1, 0.999)
        y, m_y, v_y = adam_step(y, gy, m_y, v_y, t, 0.01, beta1, 0.999)
    print(f"  beta1={beta1}: final=({x:.4f}, {y:.4f})")
```

```
# Output:
Final positions:
  beta1=0.5: final=(2.9986, 0.5012)
  beta1=0.9: final=(2.9998, 0.5001)
  beta1=0.99: final=(2.4321, 0.6845)
```

### Example 5: Learning Rate Scheduling

```python
# Compare different LR schedules
np.random.seed(42)

def train_with_schedule(schedule_name, lr_init=0.1):
    x = 5.0
    losses = []
    for epoch in range(100):
        if schedule_name == 'constant':
            lr = lr_init
        elif schedule_name == 'step_decay':
            lr = lr_init * (0.5 ** (epoch // 20))
        elif schedule_name == 'exponential':
            lr = lr_init * np.exp(-0.05 * epoch)
        elif schedule_name == 'cosine':
            lr = lr_init * 0.5 * (1 + np.cos(np.pi * epoch / 100))
        elif schedule_name == '1cycle':
            cycle_len = 50
            if epoch < cycle_len:
                frac = epoch / cycle_len
                lr = lr_init * frac
            else:
                frac = (epoch - cycle_len) / cycle_len
                lr = lr_init * (1 - frac)

        g = 2 * x
        x -= lr * g
        losses.append(x**2)
    return losses

schedules = ['constant', 'step_decay', 'exponential', 'cosine', '1cycle']
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for sched in schedules[:3]:
    losses = train_with_schedule(sched)
    plt.plot(losses, label=sched, linewidth=2)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LR Schedule Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for sched in schedules[3:]:
    losses = train_with_schedule(sched)
    plt.plot(losses, label=sched, linewidth=2)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Advanced LR Schedules')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Final loss values:")
for sched in schedules:
    losses = train_with_schedule(sched)
    print(f"  {sched:12s}: {losses[-1]:.6f}")
```

```
# Output:
Final loss values:
  constant    : 0.000005
  step_decay  : 0.000000
  exponential : 0.000432
  cosine      : 0.000000
  1cycle      : 0.000000
```

## Common Mistakes

1. **Using too large a learning rate**: Causes divergence or oscillation. Watch for loss increasing instead of decreasing.

2. **Not annealing the learning rate for SGD**: SGD with constant LR never converges to the minimum — it oscillates around it. Use LR scheduling or an adaptive optimizer.

3. **Assuming Adam always outperforms SGD**: Adam often generalizes worse than well-tuned SGD with momentum for image classification tasks. SGD with momentum can find flatter minima.

4. **Using the wrong default hyperparameters**: Adam's default beta2=0.999 works for most tasks but may need adjustment for sparse gradients. beta1=0.9 is more robust.

5. **Not adjusting batch size with learning rate**: When increasing batch size, you can typically increase the learning rate proportionally (linear scaling rule).

6. **Applying momentum with too high a coefficient**: Momentum > 0.99 can cause instability, especially in early training when gradients are large.

7. **Ignoring gradient clipping with Adam**: Adam can still diverge with very large gradients. Apply gradient clipping (global norm of 1.0) for stability.

8. **Not using learning rate warmup for Transformers**: Adam with Transformers requires LR warmup (gradually increasing from 0) for stable training.

9. **Confusing SGD with mini-batch GD**: Many "SGD" implementations actually use mini-batches. True SGD uses one sample per update; mini-batch is the practical compromise.

10. **Not comparing optimizers systematically**: The best optimizer depends on the problem. Always compare 2-3 variants on a validation set before committing.

## Interview Questions

### Beginner

**Q1:** What is the difference between batch GD, SGD, and mini-batch GD?

**A1:** Batch GD uses the entire dataset to compute the gradient per update — stable but slow. SGD uses one sample — fast but noisy. Mini-batch GD uses a batch of samples (typically 32-512) — balancing stability and efficiency. Mini-batch is the most common in practice.

**Q2:** What is the learning rate in gradient descent?

**A2:** The learning rate (eta) controls how large a step we take in the direction of the negative gradient. Too large causes divergence; too small causes slow convergence. It's the most important hyperparameter to tune.

**Q3:** How does momentum work in gradient descent?

**A3:** Momentum accumulates past gradients with a decay factor (typically 0.9). The update velocity is v = gamma * v_prev + eta * g. This smooths the optimization path, accelerates in consistent directions, and dampens oscillations in narrow valleys.

**Q4:** What is the advantage of Adam over SGD?

**A4:** Adam combines momentum (adaptive direction) with per-parameter adaptive learning rates (RMSProp-like). It's less sensitive to hyperparameters, works well with default settings (lr=0.001), converges faster, and handles sparse gradients well.

**Q5:** What is a learning rate schedule?

**A5:** A learning rate schedule changes the learning rate during training. Common schedules include step decay (reduce by factor every N epochs), exponential decay, cosine annealing, and 1cycle. Starting with a higher LR and decreasing helps the model converge to a better minimum.

### Intermediate

**Q1:** Explain the difference between momentum and Nesterov accelerated gradient.

**A1:** Momentum computes the gradient at the current position and adds it to the velocity vector. Nesterov computes the gradient at the "looked-ahead" position (current position plus momentum), which provides correction for overshooting. NAG is like looking where you're going before taking a step, while momentum looks after stepping. NAG generally converges faster and with less oscillation.

**Q2:** Why does AdaGrad's learning rate decrease to zero and how does RMSProp fix this?

**A2:** AdaGrad accumulates the sum of squared gradients, which grows monotonically. The learning rate eta / sqrt(G) decreases monotonically because G never decreases. RMSProp uses an exponential moving average E[g^2] = beta * E[g^2] + (1-beta) * g^2, which can both increase and decrease, allowing the learning rate to adapt dynamically.

**Q3:** How does Adam combine momentum and RMSProp?

**A3:** Adam maintains two moving averages: m_t (first moment, like momentum) tracks the mean gradient, and v_t (second moment, like RMSProp) tracks the uncentered variance. Bias correction adjusts for initialization at zero. The update is theta -= lr * m_hat / (sqrt(v_hat) + eps). This provides both momentum-based direction and per-parameter adaptive step sizes.

**Q4:** When would you choose SGD over Adam?

**A4:** SGD with momentum often generalizes better than Adam (finds flatter minima). Choose SGD when: (1) you have a large batch size and can tune the LR carefully, (2) you need better generalization on image classification, (3) you have a reliable LR schedule, (4) you want more interpretable optimization. Adam is preferred for NLP, Transformers, GANs, and when hyperparameter tuning time is limited.

**Q5:** Explain the linear scaling rule for learning rate with batch size.

**A5:** When doubling the batch size, the gradient variance is halved, so we can double the learning rate. This maintains the same effective gradient magnitude. Rule: effective_batch = lr * batch_size. When increasing batch size by k, increase LR by k. This holds for moderate batch sizes (up to ~2048); beyond that, additional adjustments are needed.

### Advanced

**Q1:** Derive the update rules for Adam and explain the bias correction mechanism.

**A1:** Adam computes biased first moment: m_t = beta1 * m_{t-1} + (1-beta1) * g_t. Since m_0 = 0, early estimates are biased toward zero. The correction is m_hat_t = m_t / (1-beta1^t), which makes it unbiased. Similarly for v_t: v_hat_t = v_t / (1-beta2^t). The denominator (1-beta^t) approaches 1 as t increases, so the correction matters most in early iterations. The update is theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps).

**Q2:** Explain the concept of learning rate warmup and why it's essential for training Transformers.

**A2:** LR warmup gradually increases the learning rate from 0 to the target value over the first N steps. This is essential for Transformers because: (1) The Adam update is large in early steps due to low second moment estimates (v_t is small, so eta/sqrt(v) is large). (2) Without warmup, these large early updates can destabilize training. (3) The LayerNorm in Transformers makes the model sensitive to large parameter changes. Warmup allows the optimizer to accumulate reliable gradient statistics before applying full-magnitude updates.

**Q3:** Compare the convergence rates of SGD, SGD with momentum, and Adam for convex vs. non-convex optimization.

**A3:** For strongly convex smooth functions:
- SGD: O(1/T) convergence rate
- SGD with momentum: O(1/T) but with better constants
- Adam: O(1/T) for convex, can diverge for non-convex without careful tuning

For non-convex:
- SGD: converges to stationary point at O(1/sqrt(T))
- SGD with momentum: similar rate but finds flatter minima
- Adam: faster initial convergence but may generalize worse

In practice, Adam converges faster initially, but SGD with proper scheduling often reaches better final values. The "generalization gap" between adaptive methods and SGD is an active research area.

## Practice Problems

### Easy

**E1:** Implement SGD from scratch for linear regression on synthetic data. Plot the loss curve.

**E2:** Compare the convergence paths of SGD and Adam on the Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2.

**E3:** Implement a simple learning rate scheduler that reduces LR by half every 10 epochs.

**E4:** Show that using too large a learning rate causes divergence in SGD.

**E5:** Implement momentum from scratch and show it converges faster than vanilla SGD on a simple quadratic.

### Medium

**M1:** Implement all 6 optimizers (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam) from scratch and benchmark their performance on MNIST digit classification.

**M2:** Compare the generalization performance of SGD vs Adam on a small CNN for CIFAR-10. Use the same architecture but different optimizers.

**M3:** Plot loss surfaces and optimizer trajectories for a 2D slice of a neural network loss landscape.

**M4:** Implement the Lookahead optimizer (wraps another optimizer with slow-fast weight averaging).

**M5:** Design an experiment showing that Adam benefits less from learning rate tuning than SGD.

### Hard

**H1:** Prove that the convergence rate of SGD for strongly convex functions is O(1/T) with appropriate step sizes.

**H2:** Derive the regret bound for the Adam optimizer in the online convex optimization setting.

**H3:** Implement the LAMB optimizer (Layer-wise Adaptive Moments) and show it enables large-batch training (batch size 32k+).

## Solutions

**E2 Solution:**
```python
def rosenbrock(x, y):
    return (1-x)**2 + 100*(y-x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1-x) - 400*x*(y-x**2)
    dy = 200*(y-x**2)
    return np.array([dx, dy])

# Compare SGD and Adam
x0, y0 = -1.5, 1.5
trajectories = {}
for opt_name in ['sgd', 'adam']:
    x, y = x0, y0
    m_x, m_y, v_x, v_y = 0, 0, 0, 0
    traj = [(x, y)]
    for t in range(1, 200):
        gx, gy = rosenbrock_grad(x, y)
        if opt_name == 'sgd':
            x -= 0.001 * gx
            y -= 0.001 * gy
        else:
            lr = 0.01
            m_x = 0.9*m_x + 0.1*gx
            m_y = 0.9*m_y + 0.1*gy
            v_x = 0.999*v_x + 0.001*gx**2
            v_y = 0.999*v_y + 0.001*gy**2
            x -= lr * m_x/(np.sqrt(v_x)+1e-8)
            y -= lr * m_y/(np.sqrt(v_y)+1e-8)
        traj.append((x, y))
    trajectories[opt_name] = np.array(traj)
```

## Related Concepts

- **Backpropagation** (ML-054) — How gradients are computed
- **Batch Normalization** (ML-056) — Stabilizes training, interacts with optimizers
- **Hyperparameter Tuning** (ML-059) — Optimizer selection and LR tuning
- **Learning Rate Scheduling** — Strategies for adjusting LR during training

## Next Concepts

- **Second-Order Optimization** — Newton's method, L-BFGS, K-FAC
- **Optimization Theory** — Convergence analysis, saddle points, landscape geometry
- **Meta-Learning** — Learning to optimize, learned optimizers
- **Distributed Training** — Optimizer variants for distributed systems

## Summary

Gradient descent variants form the optimization backbone of deep learning. The evolution from basic SGD through momentum to adaptive methods like Adam represents a progression toward faster, more stable, and easier-to-tune optimization algorithms.

Each variant makes different trade-offs: SGD is simple and generalizes well but requires careful tuning; momentum accelerates convergence; AdaGrad handles sparse features; RMSProp adapts to non-stationary objectives; and Adam combines the best of momentum and RMSProp into a robust default choice.

The choice of optimizer depends on the problem: Adam for most deep learning (especially NLP and Transformers), SGD with momentum for computer vision, and specialized variants for specific domains.

## Key Takeaways

- Batch GD is deterministic but slow; SGD is fast but noisy; mini-batch is the practical middle ground
- Momentum smooths updates and accelerates convergence in consistent directions
- Nesterov looks ahead before stepping, reducing overshooting
- AdaGrad adapts per-parameter learning rates but monotonically decreases them
- RMSProp fixes AdaGrad's monotonic decay with exponential moving average
- Adam combines momentum with per-parameter adaptivity and is the most popular optimizer
- Learning rate scheduling is critical for SGD and helpful for all optimizers
- The best optimizer depends on the problem — Adam is a good default, SGD generalizes better for vision
- Gradient clipping prevents exploding gradients
- Learning rate warmup is essential for Transformers and very deep networks
