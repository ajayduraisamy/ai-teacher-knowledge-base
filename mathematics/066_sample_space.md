# Concept: Sample Space

## Concept ID

MATH-066

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- Define a sample space as the set of all possible outcomes of a random experiment
- Distinguish between discrete and continuous sample spaces
- Distinguish between finite and infinite sample spaces
- Construct sample spaces for common random experiments
- Understand how sample spaces relate to events and probability
- Recognise sample spaces in machine learning contexts

## Prerequisites

- Basic set theory notation
- Understanding of probability axioms (MATH-065)
- Elementary counting principles

## Definition

A **sample space**, denoted by $S$ or $\Omega$, is the set of all possible outcomes of a random experiment. Every outcome in a sample space is mutually exclusive (only one outcome can occur per trial) and collectively exhaustive (every possible outcome is included).

Formally, if we perform an experiment whose outcome is subject to chance, the sample space $S$ contains every distinct result that could possibly occur. Subsets of the sample space are called events, and probabilities are assigned to events.

## Intuition

The sample space is the universe of possibilities. When you flip a coin, two things can happen: heads or tails. The sample space is $\{\text{heads}, \text{tails}\}$. When you roll a die, six things can happen, and the sample space is $\{1, 2, 3, 4, 5, 6\}$.

Think of the sample space as a complete catalogue of every possible outcome before the experiment begins. No outcome outside this catalogue is possible. The probability of the entire sample space is always 1 because one of the outcomes must occur.

## Why This Concept Matters

The sample space is the foundation on which all of probability is built. Every event is a subset of the sample space, and every probability calculation references the sample space. Without a clearly defined sample space, probability statements are ambiguous. In machine learning, understanding the sample space is crucial for defining what a model can output, what states an agent can encounter, and over what set of possibilities we quantify uncertainty.

## Historical Background

The concept of a sample space emerged gradually as probability theory matured. Early probability problems (Pascal, Fermat, 1654) implicitly used sample spaces — they listed all possible outcomes of dice games and card games. The formal definition of a sample space as a set of outcomes is due to **Andrey Kolmogorov** (1933), who placed probability on a rigorous set-theoretic foundation in his monograph *Foundations of the Theory of Probability*. Kolmogorov's framework defined the sample space as the fundamental building block, with events as subsets and probability as a measure on those subsets.

## Real World Examples

1. **Quality Control** — A factory tests light bulbs. The experiment is "test a bulb until it fails." The sample space is $\{1, 2, 3, \dots\}$ (the number of hours until failure), a countably infinite discrete space. Quality engineers use this to model failure rates.

2. **Weather Prediction** — The high temperature tomorrow could be any real number in a plausible range, say $[-10, 50]$ degrees Celsius. This is an uncountably infinite continuous sample space.

3. **Stock Market** — The closing price of a stock tomorrow relative to today could increase, decrease, or stay the same. The sample space is $\{\text{up}, \text{down}, \text{unchanged}\}$, a finite discrete space.

4. **Medical Diagnosis** — A patient's blood type is one of $\{A, B, AB, O\}$. This finite sample space underlies blood transfusion compatibility calculations.

5. **Network Traffic** — The number of packets arriving at a router in one second could be any non-negative integer $\{0, 1, 2, \dots\}$, a countably infinite discrete sample space used in queueing theory.

## AI/ML Relevance

1. **Output Spaces of Classifiers** — A classifier's output space corresponds to its sample space. A binary classifier has sample space $\{0, 1\}$ or $\{\text{negative}, \text{positive}\}$. A digit classifier (MNIST) has sample space $\{0, 1, 2, \dots, 9\}$. A multi-label classifier has a more complex sample space consisting of all subsets of labels.

2. **Reinforcement Learning State Spaces** — In reinforcement learning, the state space is the sample space of all possible configurations the agent can encounter. For a chess-playing agent, the state space is the set of all legal board positions. For a robotics agent, the state space is continuous (joint angles, positions, velocities).

3. **Action Spaces** — The set of all possible actions an agent can take forms a sample space. Discrete action spaces (up/down/left/right in gridworld) and continuous action spaces (torque values in robot control) correspond to discrete and continuous sample spaces.

4. **Generative Models** — Generative models learn to produce samples that resemble training data. The sample space is the space of all possible outputs (e.g., all 28×28 pixel images for MNIST generation). Understanding the structure of this space is essential for evaluating generative model quality.

5. **Bayesian Hypothesis Testing** — The sample space of hypotheses $\{H_1, H_2, \dots\}$ defines what we can conclude. In Bayesian A/B testing, the sample space of possible effect sizes is continuous, and we compute posterior probabilities over this space.

## Mathematical Explanation

### Types of Sample Spaces

**Discrete Sample Spaces**: The outcomes can be counted. If $S$ is finite or countably infinite, it is discrete.

- Finite: $S = \{1, 2, 3, 4, 5, 6\}$ (die roll)
- Countably Infinite: $S = \{0, 1, 2, 3, \dots\}$ (number of customers arriving)

**Continuous Sample Spaces**: The outcomes form a continuum, typically an interval of real numbers. These are uncountably infinite.

- $S = [0, \infty)$ (waiting time)
- $S = \mathbb{R}$ (temperature)
- $S = [0, 1]$ (probability value)

### Sample Space Cardinality

The size of a sample space is called its cardinality. Finite sample spaces have $|S| < \infty$. Infinite sample spaces can be countably infinite (like the natural numbers) or uncountably infinite (like the real numbers). The type of sample space determines whether we use a probability mass function (PMF, for discrete) or a probability density function (PDF, for continuous).

### Product Sample Spaces

When an experiment consists of multiple stages, the overall sample space is the Cartesian product of the individual sample spaces. For flipping a coin twice:
$$
S = \{\text{H}, \text{T}\} \times \{\text{H}, \text{T}\} = \{\text{HH}, \text{HT}, \text{TH}, \text{TT}\}
$$

For rolling two dice:
$$
S = \{1,2,3,4,5,6\} \times \{1,2,3,4,5,6\}
$$
which has $6 \times 6 = 36$ outcomes.

## Formula(s)

1. **Sample Space as a Set**:
   $$
   S = \{\text{all possible outcomes of a random experiment}\}
   $$

2. **Probability of Sample Space**:
   $$
   P(S) = 1
   $$

3. **Product Sample Space** (for independent stages):
   $$
   S = S_1 \times S_2 \times \dots \times S_n = \{(s_1, s_2, \dots, s_n) \mid s_i \in S_i\}
   $$

4. **Cardinality of Product Space** (finite case):
   $$
   |S| = |S_1| \times |S_2| \times \dots \times |S_n|
   $$

## Properties

1. **Exhaustiveness**: Every possible outcome is included in $S$.
2. **Mutual Exclusivity of Outcomes**: Only one outcome occurs per trial.
3. **Total Probability**: $P(S) = 1$.
4. **Empty Event**: $P(\emptyset) = 0$, the impossible event.
5. **Countability**: Sample spaces may be finite, countably infinite, or uncountably infinite.
6. **Product Structure**: Multi-stage experiments yield product sample spaces.
7. **Defines the Universe**: Every event is a subset of $S$; nothing outside $S$ is possible.

## Step-by-Step Worked Examples

### Example 1: Single Coin Flip

**Problem**: Define the sample space for flipping a fair coin once.

**Solution**:

Step 1: Identify all possible outcomes. A coin can land on either heads (H) or tails (T).

Step 2: Write the sample space:
$$
S = \{H, T\}
$$

Step 3: Note cardinality: $|S| = 2$.

Step 4: Verify properties. $P(S) = P(H) + P(T) = 0.5 + 0.5 = 1$.

### Example 2: Rolling Two Dice

**Problem**: Define the sample space for rolling a red die and a blue die. How many outcomes are there?

**Solution**:

Step 1: Each die individually has sample space $\{1, 2, 3, 4, 5, 6\}$.

Step 2: The combined experiment is the Cartesian product:
$$
S = \{1, 2, 3, 4, 5, 6\} \times \{1, 2, 3, 4, 5, 6\}
$$

Step 3: List all outcomes as ordered pairs (red, blue). For example, $(3, 5)$ means red shows 3 and blue shows 5.

Step 4: Compute cardinality: $|S| = 6 \times 6 = 36$.

Step 5: Verify. There are 36 equally likely outcomes if both dice are fair.

### Example 3: Drawing Cards Without Replacement

**Problem**: Define the sample space for drawing two cards from a standard 52-card deck without replacement.

**Solution**:

Step 1: For the first draw, there are 52 possible cards.

Step 2: For the second draw, since we do not replace the first card, there are 51 remaining possibilities.

Step 3: The sample space consists of all ordered pairs (first card, second card):
$$
S = \{(c_1, c_2) \mid c_1 \in \text{deck}, c_2 \in \text{deck} \setminus \{c_1\}\}
$$

Step 4: Compute cardinality: $|S| = 52 \times 51 = 2,652$.

Step 5: If order does not matter, the sample space for unordered draws has $\binom{52}{2} = 1,326$ outcomes.

### Example 4: Continuous Sample Space

**Problem**: A bus arrives at a stop between 8:00 AM and 8:30 AM uniformly at random. Define the sample space for the arrival time in minutes past 8:00 AM.

**Solution**:

Step 1: The arrival time can be any real number between 0 and 30.

Step 2: Write the sample space:
$$
S = [0, 30] = \{t \in \mathbb{R} \mid 0 \leq t \leq 30\}
$$

Step 3: This is an uncountably infinite continuous sample space.

Step 4: The probability of any exact time (e.g., arriving at exactly 8:10:00.000...) is 0, but the probability of arriving in any subinterval $[a, b] \subseteq [0, 30]$ is $\frac{b-a}{30}$.

### Example 5: Multi-Stage Experiment (Coin Flips Until Heads)

**Problem**: Define the sample space for flipping a fair coin repeatedly until heads appears for the first time.

**Solution**:

Step 1: Consider the possible sequences. We could get heads on the first flip (H), tails then heads (TH), tails, tails, then heads (TTH), and so on.

Step 2: The sample space is:
$$
S = \{H, TH, TTH, TTTH, \dots\}
$$

Step 3: This is a countably infinite discrete sample space. The outcome TTT...H (with $n$ tails followed by a head) corresponds to the sequence where heads appears on the $(n+1)$-th flip.

Step 4: Infinite sample spaces require special care. Despite being infinite, the probabilities of all outcomes sum to 1:
$$
\sum_{n=0}^{\infty} P(T^n H) = \sum_{n=0}^{\infty} \left(\frac{1}{2}\right)^{n+1} = \frac{1/2}{1-1/2} = 1
$$

## Visual Interpretation

A discrete sample space can be visualised as a collection of points (outcomes) within a boundary representing $S$. For a die roll, picture six equally spaced dots labelled 1 through 6. For two dice, imagine a 6×6 grid where each cell is an outcome.

A continuous sample space is visualised as a region of the real line or a higher-dimensional space. The interval $[0, 1]$ is a line segment; the set of all points in a 2D region (e.g., where a dart lands on a board) is an area. Probability is then proportional to area (or length, volume).

Tree diagrams are especially useful for multi-stage experiments. Each branch represents a possible outcome at that stage, and the leaves represent the complete outcomes in the product sample space.

## Common Mistakes

1. **Omitting outcomes**: A sample space must include every possible outcome. Forgetting that a coin could land on its edge would miss a valid (though unlikely) outcome.

2. **Confusing sample space with event**: The sample space is the set of all outcomes, not a particular subset. Events are subsets of the sample space.

3. **Assuming equally likely outcomes**: The sample space definition does not require equally likely outcomes. The sample space $\{\text{rain}, \text{no rain}\}$ is valid even if rain is less likely than no rain.

4. **Incorrect cardinality for multi-stage experiments**: For flipping 3 coins, $|S| = 2^3 = 8$, not $3 \times 2 = 6$. The product rule applies multiplicatively, not additively.

5. **Confusing with/without replacement**: Drawing two cards with replacement has $52 \times 52 = 2,704$ outcomes. Without replacement, it has $52 \times 51 = 2,652$. Neglecting the difference leads to incorrect cardinalities.

6. **Treating continuous sample spaces as discrete**: In a continuous sample space, individual points have probability 0. Treating them as discrete (assigning positive probability to every point) would violate the probability axioms.

7. **Not considering order**: When outcomes are ordered (e.g., drawing two cards, first and second), the sample space includes ordered pairs. When order does not matter, the space is different (combinations rather than permutations).

## Interview Questions

### Beginner

1. **Q**: What is a sample space?
   **A**: The sample space $S$ is the set of all possible outcomes of a random experiment. Every possible result of the experiment belongs to $S$.

2. **Q**: What is the sample space for rolling a single six-sided die?
   **A**: $S = \{1, 2, 3, 4, 5, 6\}$. It has 6 outcomes.

3. **Q**: What is the sample space for flipping a coin three times?
   **A**: $S = \{HHH, HHT, HTH, THH, HTT, THT, TTH, TTT\}$, with $2^3 = 8$ outcomes.

4. **Q**: What does it mean for a sample space to be continuous?
   **A**: A continuous sample space contains a continuum of outcomes, typically an interval of real numbers. Individual points have probability 0; only intervals have positive probability.

5. **Q**: Why must $P(S) = 1$?
   **A**: Because the sample space contains every possible outcome, and one of them must occur. The probability that something in $S$ happens is certain, so $P(S) = 1$.

### Intermediate

1. **Q**: Describe the sample space for measuring the height of a randomly selected adult.
   **A**: Height is a continuous quantity. A reasonable sample space is $S = [0, 3]$ metres (or $[0, \infty)$), an uncountably infinite continuous space.

2. **Q**: How does the sample space for drawing two cards with replacement differ from without replacement?
   **A**: With replacement: $|S| = 52 \times 52 = 2,704$. Without replacement: $|S| = 52 \times 51 = 2,652$. The sample space without replacement excludes outcomes where the same card appears twice.

3. **Q**: In reinforcement learning, what is the relationship between the state space and the sample space?
   **A**: The state space in RL is the sample space of all configurations the agent can encounter. Each state is a possible outcome of the environment's dynamics. Together with the action space, it forms the product sample space for state-action pairs.

4. **Q**: Can a sample space be infinite? Give an example relevant to machine learning.
   **A**: Yes. In a Poisson process modelling webpage visits, the number of visitors in an hour is any non-negative integer $\{0, 1, 2, \dots\}$, a countably infinite sample space.

5. **Q**: What is a product sample space and when does it arise?
   **A**: A product sample space arises when an experiment consists of multiple independent stages. It is the Cartesian product of each stage's sample space. For example, rolling two dice yields $S = \{1,\dots,6\} \times \{1,\dots,6\}$.

### Advanced

1. **Q**: Explain why individual points in a continuous sample space must have probability 0, and how this relates to the additivity axiom.
   **A**: If each point in $[0, 1]$ had positive probability $\epsilon > 0$, the sum over all points (uncountably many) would be infinite, exceeding $P(S) = 1$. Countable additivity forces individual points to have probability 0 in continuous spaces.

2. **Q**: How does the choice of sample space affect the definition of probability in a machine learning classification problem?
   **A**: The sample space determines the output format and the loss function. For binary classification, $S = \{0,1\}$ yields Bernoulli likelihood. For multi-class with $K$ classes, $S = \{1,\dots,K\}$ yields categorical likelihood. The softmax function maps real-valued logits to probabilities over this $K$-element sample space.

3. **Q**: Describe how the sample space concept extends to stochastic processes like Gaussian processes.
   **A**: For a Gaussian process, the sample space is a function space — the set of all possible functions $f: \mathcal{X} \to \mathbb{R}$ consistent with the GP prior. Each outcome is an entire function. This is an infinite-dimensional sample space. Probability is defined over function-space using the GP's mean and covariance functions.

## Practice Problems

### Easy

1. Write the sample space for the gender of a newborn baby.

2. Write the sample space for the outcome of spinning a spinner labelled A, B, C, D.

3. A coin is flipped and a die is rolled simultaneously. Write the sample space and compute its cardinality.

4. Write the sample space for the number of heads when flipping 2 coins (considering unordered outcomes).

5. A bag contains red, blue, and green marbles. One marble is drawn. Write the sample space.

### Medium

1. A family has three children. Write the sample space for the genders (B for boy, G for girl) in birth order.

2. Two cards are drawn from a 52-card deck with replacement. How many outcomes are in the sample space?

3. A random number between 0 and 1 is generated. Write the sample space and classify it as discrete or continuous.

4. An experiment consists of rolling a die until a 6 appears. Write the sample space for the number of rolls needed.

5. A combination lock has 3 dials, each labelled 0-9. Write the sample space for the combination and compute its size.

### Hard

1. In the Monty Hall problem, a contestant chooses one of three doors, then the host (who knows what is behind each door) opens a door with a goat, and the contestant may switch. Define the complete sample space for this experiment, including the initial car placement, the contestant's initial choice, the host's action, and the final outcome if the contestant switches.

2. A point is chosen uniformly at random from a circle of radius 1. Describe the sample space. What is the probability that the point lies within a distance 0.5 of the centre?

3. Prove that for any countably infinite sample space, it is impossible to assign equal positive probability to every outcome.

## Solutions

### Easy Solutions

**Solution 1**: $S = \{B, G\}$ (boy or girl), assuming binary classification for simplicity.

**Solution 2**: $S = \{A, B, C, D\}$, cardinality 4.

**Solution 3**: $S = \{H, T\} \times \{1, 2, 3, 4, 5, 6\}$. Cardinality $= 2 \times 6 = 12$.

**Solution 4**: Considering unordered counts: $S = \{0, 1, 2\}$ heads. (If order matters: $\{HH, HT, TH, TT\}$.)

**Solution 5**: $S = \{\text{red}, \text{blue}, \text{green}\}$, cardinality 3.

### Medium Solutions

**Solution 1**: $S = \{BBB, BBG, BGB, GBB, BGG, GBG, GGB, GGG\}$, cardinality $2^3 = 8$.

**Solution 2**: $|S| = 52 \times 52 = 2,704$.

**Solution 3**: $S = [0, 1]$, the interval of real numbers between 0 and 1. This is a continuous (uncountably infinite) sample space.

**Solution 4**: $S = \{1, 2, 3, 4, \dots\}$, a countably infinite discrete sample space.

**Solution 5**: $S = \{0,1,\dots,9\} \times \{0,1,\dots,9\} \times \{0,1,\dots,9\}$, $|S| = 10^3 = 1,000$.

### Hard Solutions

**Solution 1**: Let car be behind door $C \in \{1,2,3\}$, contestant initially picks $P \in \{1,2,3\}$, host opens $H \in \{1,2,3\} \setminus \{C, P\}$, contestant switches to $F \in \{1,2,3\} \setminus \{P, H\}$. The full sample space includes all valid $(C, P, H, F)$ tuples satisfying the rules. The outcome if switching is determined by whether $F = C$.

**Solution 2**: The sample space is $S = \{(x, y) \in \mathbb{R}^2 \mid x^2 + y^2 \leq 1\}$, all points in the unit disk. The event $E = \{\text{distance} \leq 0.5\}$ is the disk of radius 0.5. $P(E) = \frac{\text{area}(E)}{\text{area}(S)} = \frac{\pi (0.5)^2}{\pi (1)^2} = 0.25$.

**Solution 3**: Suppose each outcome $o_i$ has probability $p > 0$. Then $\sum_{i=1}^{\infty} P(o_i) = \sum_{i=1}^{\infty} p = \infty$, which violates the axiom $P(S) = 1$. Therefore, equal positive probability cannot be assigned to countably infinitely many outcomes. Probabilities must form a convergent series summing to 1 (e.g., $P(o_i) = 1/2^i$).

## Related Concepts

- **Probability (MATH-065)**: The measure defined on the sample space
- **Event (MATH-067)**: A subset of the sample space
- **Random Variable (MATH-070)**: A function from the sample space to real numbers
- **Probability Distribution (MATH-071)**: A function describing probabilities of events
- **Set Theory**: The mathematical language for describing sample spaces
- **Cartesian Product**: Used to construct product sample spaces

## Next Concepts

- **Event**: Subsets of the sample space that we assign probabilities to (MATH-067)
- **Random Variable**: Mapping outcomes to numerical values (MATH-070)
- **Probability Distribution**: Assigning probabilities systematically (MATH-071)

## Summary

The sample space $S$ is the foundational set in probability theory, containing every possible outcome of a random experiment. Sample spaces can be discrete (finite or countably infinite) or continuous (uncountably infinite). The probability of the entire sample space is always 1. Multi-stage experiments produce product sample spaces whose cardinality is the product of the stage cardinalities. In machine learning, sample spaces define output spaces of classifiers, state spaces in reinforcement learning, and the domain of generative models. Correctly identifying the sample space is the first and most critical step in any probability problem.

## Key Takeaways

- The sample space $S$ is the set of all possible outcomes of an experiment
- $P(S) = 1$ by the normalisation axiom
- Sample spaces are classified as discrete (finite or countably infinite) or continuous (uncountably infinite)
- Product sample spaces model multi-stage experiments via Cartesian products
- In continuous sample spaces, individual outcomes have probability 0
- The type of sample space determines whether PMFs or PDFs are used
- Machine learning relies on sample spaces for output definitions, state spaces, and action spaces
- Always explicitly define the sample space before computing probabilities
- Tree diagrams help visualise multi-stage sample spaces
- The cardinality of a product space is the product of individual cardinalities
