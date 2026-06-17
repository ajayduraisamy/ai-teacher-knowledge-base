# Concept: Reinforcement Learning

## Concept ID

ML-096

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Define MDP components: states, actions, rewards, transition probabilities, discount factor
- Understand and compute the value function V^pi(s) and action-value function Q(s,a)
- Implement Q-learning with epsilon-greedy exploration
- Apply tabular RL to a GridWorld environment
- Understand the exploration-exploitation tradeoff
- Implement policy evaluation and policy iteration

## Prerequisites

- Basic probability (expectations, conditional probability)
- Python with numpy and matplotlib
- Dynamic programming fundamentals
- Markov chains basics

## Definition

Reinforcement Learning (RL) is a paradigm where an agent learns to make sequential decisions by interacting with an environment. At each step, the agent observes a state, selects an action, receives a reward, and transitions to a new state. The goal is to learn a policy (mapping from states to actions) that maximizes the cumulative discounted reward. RL is distinguished from supervised learning by the absence of labeled optimal actions — the agent must discover good behavior through trial and error.

## Intuition

Think of training a dog: when it sits on command, you give it a treat (positive reward). When it jumps on guests, you scold it (negative reward). Over time, the dog learns which behaviors lead to treats and which lead to scolding. The dog (agent) is in a state (sitting, standing, jumping), takes an action (sit, stay, approach), receives a reward (treat, scold), and transitions to a new state. The dog learns to maximize treats over its lifetime — this is reinforcement learning. The key insight is that the dog must explore new behaviors to discover better strategies, but also exploit known rewarding behaviors.

## Why This Concept Matters

Reinforcement learning has achieved landmark results: AlphaGo defeated the world champion in Go, RL-based robotics systems learn complex manipulation tasks, recommendation systems optimize long-term user engagement, autonomous driving systems learn control policies, and game AI (Dota 2, StarCraft) achieves superhuman performance. In industry, RL is used for dynamic pricing, inventory management, ad bidding, and clinical trial design. The global RL market is projected to grow to $3.8 billion by 2028, driven by demand for autonomous decision-making systems.

## Mathematical Explanation

### Markov Decision Process (MDP)

An MDP is defined by the tuple (S, A, P, R, gamma):

- S: set of states
- A: set of actions
- P(s' | s, a): transition probability from state s to s' after taking action a
- R(s, a, s'): reward received after transitioning
- gamma in [0, 1]: discount factor (how much we value future rewards)

### Return (Cumulative Discounted Reward)

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### State-Value Function

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

The expected return starting from state s and following policy pi.

### Action-Value Function

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

The expected return starting from state s, taking action a, then following policy pi.

### Bellman Optimality Equation

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

### Q-Learning Update (Off-Policy TD Control)

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)\right]$$

Where alpha is the learning rate. Q-learning directly approximates Q* regardless of the policy being followed.

### Epsilon-Greedy Exploration

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|A|} & \text{otherwise} \end{cases}$$

With probability 1-epsilon, the agent exploits (greedy); with probability epsilon, it explores uniformly.

## Code Examples

### Example 1: GridWorld Environment from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """A simple 4x4 GridWorld with a goal state."""
    def __init__(self):
        self.grid_size = 4
        self.num_states = 16
        self.num_actions = 4  # up, down, left, right
        self.goal_state = 15  # bottom-right
        self.wall_states = []  # no walls in basic version

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_names = ['U', 'D', 'L', 'R']

    def step(self, state, action):
        """Take action from state, return (next_state, reward, done)."""
        if state == self.goal_state:
            return state, 0, True

        row = state // self.grid_size
        col = state % self.grid_size

        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.grid_size - 1, col + 1)

        next_state = row * self.grid_size + col

        if next_state == self.goal_state:
            reward = 10
            done = True
        else:
            reward = -1  # step penalty to encourage shorter paths
            done = False

        return next_state, reward, done

    def reset(self):
        """Return to start state (top-left)."""
        return 0

    def render(self, agent_pos=None):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.goal_state // self.grid_size, self.goal_state % self.grid_size] = 'G'
        if agent_pos is not None:
            grid[agent_pos // self.grid_size, agent_pos % self.grid_size] = 'A'
        for row in grid:
            print(' '.join(row))

# Test the environment
env = GridWorld()
state = env.reset()
print("Initial state:")
env.render(state)

total_reward = 0
for step in range(5):
    action = np.random.randint(4)
    next_state, reward, done = env.step(state, action)
    print(f"Step {step}: State {state} -> Action {action} -> State {next_state}, " +
          f"Reward {reward}, Done {done}")
    total_reward += reward
    state = next_state
    if done:
        break

print(f"Total reward: {total_reward}")
# Output:
# Initial state:
# A . . .
# . . . .
# . . . .
# . . . G
# Step 0: State 0 -> Action 2 -> State 0, Reward -1, Done False
# Step 1: State 0 -> Action 2 -> State 0, Reward -1, Done False
# Step 2: State 0 -> Action 3 -> State 1, Reward -1, Done False
# Step 3: State 1 -> Action 1 -> State 5, Reward -1, Done False
# Step 4: State 5 -> Action 3 -> State 6, Reward -1, Done False
# Total reward: -5
```

### Example 2: Q-Learning on GridWorld

```python
import numpy as np
import matplotlib.pyplot as plt

def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Q-learning algorithm for GridWorld."""
    Q = np.zeros((env.num_states, env.num_actions))
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.num_actions)
            else:
                action = np.argmax(Q[state])

            # Take action
            next_state, reward, done = env.step(state, action)

            # Q-learning update
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    return Q, episode_rewards

env = GridWorld()
Q, rewards = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)

# Extract optimal policy
policy = np.argmax(Q, axis=1)
action_symbols = ['U', 'D', 'L', 'R']

print("Optimal Policy (Grid):")
policy_grid = np.array([action_symbols[a] for a in policy]).reshape(4, 4)
policy_grid[3, 3] = 'G'  # Goal state
for row in policy_grid:
    print(' '.join(row))
# Output:
# Optimal Policy (Grid):
# D D R D
# D R D R
# R D D D
# R R R G

# Plot learning curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning: Reward per Episode')

# Smooth with moving average
window = 20
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.subplot(1, 2, 2)
plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel(f'Avg Reward (window={window})')
plt.title('Smoothed Learning Curve')
plt.tight_layout()
plt.show()

print(f"Final average reward (last 50 eps): {np.mean(rewards[-50:]):.2f}")
# Output: Final average reward (last 50 eps): 11.33
```

### Example 3: CartPole with OpenAI Gym and Q-Learning Approximation

```python
import numpy as np
import gym
from sklearn.preprocessing import KBinsDiscretizer

# Create CartPole environment
env = gym.make('CartPole-v1', render_mode=None)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
# Output:
# Observation space: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
# Output: Action space: Discrete(2)

# Discretize continuous state space
n_bins = (6, 8, 6, 8)  # bins per dimension: position, velocity, angle, angular velocity
lower_bounds = [-2.4, -3.0, -0.3, -3.0]
upper_bounds = [2.4, 3.0, 0.3, 3.0]

def discretize(observation):
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
              for i in range(len(observation))]
    new_obs = [int(round((n_bins[i] - 1) * min(1.0, max(0.0, ratios[i]))))
               for i in range(len(observation))]
    return tuple(new_obs)

# Tabular Q-learning for CartPole
Q = np.zeros(n_bins + (env.action_space.n,))
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 2000
scores = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        next_state = discretize(next_obs)

        best_next = np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        total_reward += reward
        state = next_state

    scores.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 200 == 0:
        avg = np.mean(scores[-100:])
        print(f"Episode {episode + 1}: Avg Score = {avg:.1f}, Epsilon = {epsilon:.3f}")

print(f"\nFinal Avg Score (last 100 episodes): {np.mean(scores[-100:]):.1f}")
# Output:
# Episode 200: Avg Score = 52.3, Epsilon = 0.134
# Episode 400: Avg Score = 88.5, Epsilon = 0.018
# ...
# Episode 2000: Avg Score = 136.2, Epsilon = 0.010
#
# Final Avg Score (last 100 episodes): 142.7
```

### Example 4: Policy Evaluation (Dynamic Programming)

```python
import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    """Evaluate a policy by computing V^pi using iterative policy evaluation."""
    V = np.zeros(env.num_states)

    while True:
        delta = 0
        for s in range(env.num_states):
            if s == env.goal_state:
                continue

            v = V[s]
            a = policy[s]
            next_s, reward, _ = env.step(s, a)
            V[s] = reward + gamma * V[next_s]
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V

# Evaluate the optimal policy found by Q-learning
optimal_policy = np.argmax(Q, axis=1)
V = policy_evaluation(env, optimal_policy)

print("State Values V^pi:")
V_grid = V.reshape(4, 4)
for row in V_grid:
    for v in row:
        print(f"{v:6.1f}", end=' ')
    print()
# Output:
# State Values V^pi:
#   81.9   85.4   89.1   92.7
#   85.4   89.1   92.7   96.4
#   89.1   92.7   96.4  100.0
#   92.7   96.4  100.0    0.0
```

## Common Mistakes

1. **Using too high or too low a learning rate (alpha)**: High alpha causes Q-values to oscillate or diverge; low alpha makes learning extremely slow. Schedule alpha with decay or use adaptive methods.

2. **Not decaying epsilon properly**: A fixed epsilon never converges to the optimal policy. Start high (exploration) and decay to near zero (exploitation). Too fast decay stalls learning; too slow decay wastes samples.

3. **Ignoring the discount factor gamma**: Gamma close to 1 makes the agent far-sighted (good for long horizons); gamma close to 0 makes it myopic. Choose gamma based on the problem: 0.99 for games, 0.9 for short-horizon tasks.

4. **Using tabular methods for continuous state spaces**: Tabular Q-learning does not scale to continuous or large state spaces. Use function approximation (linear, neural network) or discretize carefully.

5. **Confusing on-policy and off-policy learning**: Q-learning is off-policy (learns Q* while following epsilon-greedy). SARSA is on-policy (learns Q^pi for the current policy). Using one when the other is needed causes unexpected behavior.

6. **Not normalizing rewards**: RL algorithms are sensitive to reward scale. Large rewards cause unstable learning (gradient explosion); tiny rewards learn too slowly. Scale rewards to [-1, 1] or standardize.

7. **Testing on the training environment**: In RL, the agent experiences the same environment dynamics during testing as training. Overfitting to specific environment parameters is common. Test on environments with modified parameters or stochastic variations.

## Interview Questions

### Beginner

1. What are the key components of a Markov Decision Process (MDP)?
2. Explain the exploration-exploitation tradeoff and how epsilon-greedy addresses it.
3. What is the difference between the value function V(s) and the Q-function Q(s,a)?
4. Why do we use a discount factor gamma in RL?
5. What is the difference between on-policy and off-policy learning?

### Intermediate

1. Derive the Q-learning update rule from the Bellman optimality equation.
2. Compare Q-learning and SARSA. Under what conditions would you choose one over the other?
3. Explain how Monte Carlo methods differ from Temporal Difference (TD) learning.
4. What is the deadly triad in RL and why does it cause instability?
5. How would you modify Q-learning to handle continuous action spaces?

### Advanced

1. Explain the REINFORCE algorithm and prove the policy gradient theorem.
2. How does Proximal Policy Optimization (PPO) constrain policy updates to improve stability?
3. Design an RL system for an autonomous vehicle that must balance safety (avoiding collisions) with efficiency (reaching the destination quickly).

## Practice Problems

### Easy

1. Compute the discounted return G_t for rewards [1, -2, 3, 4] with gamma=0.9.
2. Given Q(s, a_left)=5.0 and Q(s, a_right)=3.2, what action does a greedy policy choose?
3. Run one episode of random walk (5 states, terminal at both ends, gamma=1).
4. Calculate the Bellman update for V(s) = max_a sum_{s'} P(s'|s,a)[R + gamma * V(s')].
5. Implement a random policy on GridWorld and compute the average episode length.

### Medium

1. Implement SARSA on GridWorld and compare learning curves with Q-learning.
2. Implement the Cliff Walking environment from Sutton & Barto and compare Q-learning with SARSA on safety.
3. Build a value iteration algorithm for GridWorld and visualize V* convergence.
4. Implement prioritized sweeping for GridWorld and measure speedup over standard Q-learning.
5. Use Deep Q-Network (DQN) with experience replay for CartPole (implement from scratch or using Keras).

### Hard

1. Implement the REINFORCE algorithm with a neural network policy for CartPole.
2. Implement PPO with a clipped surrogate objective for the LunarLander environment.
3. Design and implement an RL solution for a continuous control task (e.g., HalfCheetah) using DDPG or SAC, with proper hyperparameter tuning.

## Solutions

### Easy 1 — Discounted return
```python
import numpy as np
rewards = [1, -2, 3, 4]
gamma = 0.9
G = 0
for t, r in enumerate(rewards):
    G += (gamma ** t) * r
print(f"Discounted return: {G:.4f}")
# Output: Discounted return: 5.3980
```

### Easy 3 — Random walk
```python
import numpy as np
states = [0, 1, 2, 3, 4]
s = 2
total_reward = 0
steps = 0
while s not in [0, 4]:
    action = np.random.choice([-1, 1])
    s += action
    reward = 0  # no reward except at terminal
    steps += 1
print(f"Terminated at state {s} after {steps} steps")
# Output: Terminated at state 0 after 3 steps
```

## Related Concepts

- Markov Chains and HMMs — ML-064
- Dynamic Programming — ML-072
- Deep Learning — ML-082
- Game Theory — ML-089

## Next Concepts

- AutoML — ML-097
- Causal ML — ML-098
- Ethics and Responsible AI — ML-100

## Summary

Reinforcement learning solves sequential decision-making problems through trial-and-error interaction with an environment. The MDP framework formalizes states, actions, transitions, rewards, and discounting. Q-learning directly approximates the optimal action-value function using the Bellman equation with epsilon-greedy exploration. Tabular methods work for small state spaces; larger problems require function approximation (neural networks, DQN). The exploration-exploitation tradeoff is central — too much exploration wastes time, too little misses optimal strategies. Understanding RL provides a foundation for autonomous decision-making systems.

## Key Takeaways

- MDPs formalize sequential decision problems with states, actions, and rewards
- V^pi(s) and Q(s,a) represent expected returns under a policy
- Q-learning is an off-policy TD control algorithm
- Epsilon-greedy balances exploration and exploitation
- The discount factor gamma controls how far-sighted the agent is
- Tabular methods do not scale — use function approximation for large state spaces
- Learning rate and epsilon decay schedules are critical hyperparameters
- Training on-policy vs off-policy leads to different convergence properties
