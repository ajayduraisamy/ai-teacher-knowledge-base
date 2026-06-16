# Concept: Conditional Probability

## Concept ID

MATH-068

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- Define conditional probability $P(A|B)$ as the probability of $A$ given $B$ has occurred
- Apply the multiplication rule $P(A \cap B) = P(A|B)P(B)$
- Determine independence of events using $P(A|B) = P(A)$ or $P(A \cap B) = P(A)P(B)$
- Use the law of total probability to compute unconditional probabilities
- Understand the chain rule for multiple events
- Connect conditional probability to posterior inference in machine learning

## Prerequisites

- Basic probability axioms (MATH-065)
- Sample space and event concepts (MATH-066, MATH-067)
- Basic set theory and algebra

## Definition

**Conditional probability** is the probability of event $A$ occurring given that event $B$ has already occurred. It is denoted $P(A|B)$ and defined as:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)},
$$

provided $P(B) > 0$. If $P(B) = 0$, the conditional probability is undefined (or defined as 0 by convention in some contexts).

The conditional probability $P(A|B)$ renormalises the probability of $A$ to the subset $B$ of the sample space. We are effectively asking: "Among all outcomes where $B$ occurs, what fraction also belong to $A$?"

## Intuition

Imagine you roll a die and are told the result is even. Given this information, what is the probability that the result is greater than 3? The original sample space $\{1,2,3,4,5,6\}$ shrinks to $\{2,4,6\}$ because we know the outcome is even. Among these three equally likely possibilities, $\{4,6\}$ are greater than 3. So $P(\text{greater than } 3 \mid \text{even}) = 2/3$.

Conditional probability captures the idea of updating beliefs with partial information. The occurrence of $B$ provides information that restricts the possible outcomes, and the probability of $A$ is recomputed within this restricted space.

## Why This Concept Matters

Conditional probability is at the heart of learning from data. Every time we update our beliefs based on new evidence, we are computing conditional probabilities. In machine learning, classification models output $P(y|x)$ — the conditional probability of label $y$ given input $x$. The Markov assumption in sequence models is a statement about conditional independence. Causal inference, Bayesian reasoning, and almost every advanced statistical technique rely on conditional probability. It is arguably the most important concept in applied probability.

## Historical Background

The formal definition of conditional probability was developed alongside the axiomatisation of probability. **Andrey Kolmogorov** (1933) provided the definition $P(A|B) = P(A \cap B)/P(B)$ in his axiomatic framework. However, the intuitive idea of conditional probability dates back to **Thomas Bayes** (1702–1761) and **Pierre-Simon Laplace** (1749–1827), who understood that probability must be updated in light of new evidence. The concept of statistical independence was formalised by **Abraham de Moivre** (1667–1754) and later by **Francis Galton** (1822–1911) and **Karl Pearson** (1857–1936).

## Real World Examples

1. **Medical Diagnosis** — $P(\text{disease}|\text{positive test})$ is the conditional probability that matters to patients. A positive test result updates the probability of having the disease from the base rate (prior) to a higher value (posterior).

2. **Customer Analytics** — $P(\text{purchase}|\text{email opened})$ measures the effectiveness of an email campaign. Companies compute this to determine conversion rates conditional on engagement.

3. **Weather Forecasting** — $P(\text{rain}|\text{cloudy skies})$ is higher than the unconditional probability of rain. Forecasters use conditional probabilities from historical data to make predictions.

4. **Sports** — $P(\text{Team A wins}|\text{leading at half-time})$ quantifies how momentum or advantage changes win probability. Analysts use this for in-game strategy decisions.

5. **Credit Risk** — $P(\text{default}|\text{late payment})$ helps lenders assess risk. The conditional probability of default given a missed payment is much higher than the base default rate.

## AI/ML Relevance

1. **Posterior Probability in Classification** — Every classifier that outputs probabilities is computing $P(y|x)$, the conditional probability of class $y$ given input $x$. Logistic regression models $P(y=1|x) = \sigma(w^T x)$, where $\sigma$ is the sigmoid function. Neural networks generalise this with softmax outputs.

2. **Markov Assumption** — In sequence models (HMMs, RNNs, Transformers), the Markov assumption states $P(x_t | x_{t-1}, x_{t-2}, \dots, x_1) = P(x_t | x_{t-1})$ or, more generally, limited context dependence. This conditional independence assumption makes modelling sequential data tractable.

3. **Conditional Likelihood** — Supervised learning maximises the conditional likelihood of labels given inputs: $\prod_i P(y_i | x_i; \theta)$. The loss function (negative log-likelihood) is derived from this conditional probability.

4. **Bayesian Neural Networks** — BNNs treat weights as random variables and compute $P(\theta | \mathcal{D}) \propto P(\mathcal{D} | \theta) P(\theta)$, the posterior distribution of weights conditional on the data. Predictions are made via $P(y|x, \mathcal{D}) = \int P(y|x, \theta) P(\theta|\mathcal{D}) d\theta$.

5. **Autoregressive Models** — Models like GPT generate text by factorising $P(x_1, x_2, \dots, x_n) = \prod_t P(x_t | x_{<t})$, the product of conditional probabilities of each token given previous tokens.

6. **Causal Inference** — Conditional probability is central to causal reasoning. $P(Y|do(X))$ (the probability of $Y$ given that we intervene on $X$) differs from $P(Y|X)$ (given that we observe $X$). Understanding this distinction is critical for ML fairness and decision-making.

## Mathematical Explanation

### The Multiplication Rule

Rearranging the definition of conditional probability gives the multiplication rule:
$$
P(A \cap B) = P(A|B) P(B) = P(B|A) P(A)
$$

This rule is used to compute joint probabilities from conditional ones.

### The Chain Rule (General Multiplication Rule)

For $n$ events $A_1, A_2, \dots, A_n$:
$$
P(A_1 \cap A_2 \cap \dots \cap A_n) = P(A_1) P(A_2|A_1) P(A_3|A_1 \cap A_2) \cdots P(A_n | A_1 \cap \dots \cap A_{n-1})
$$

This is the foundation of autoregressive modelling in machine learning.

### Independence

Events $A$ and $B$ are independent if any of the following equivalent conditions holds:
- $P(A|B) = P(A)$ (knowledge of $B$ does not change $A$'s probability)
- $P(B|A) = P(B)$
- $P(A \cap B) = P(A)P(B)$

Independence does not mean the events cannot occur together — it means they do not influence each other probabilistically.

### Conditional Independence

Events $A$ and $B$ are conditionally independent given $C$ if:
$$
P(A \cap B | C) = P(A|C) P(B|C)
$$

This is a weaker condition than unconditional independence and is ubiquitous in graphical models (naive Bayes, Markov random fields).

### Law of Total Probability

If $\{B_1, B_2, \dots, B_n\}$ is a partition of the sample space (mutually exclusive and exhaustive), then:
$$
P(A) = \sum_{i=1}^n P(A \cap B_i) = \sum_{i=1}^n P(A|B_i) P(B_i)
$$

This expresses the unconditional probability of $A$ as a weighted average of its conditional probabilities given each $B_i$.

## Formula(s)

1. **Definition of Conditional Probability**:
   $$
   P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
   $$

2. **Multiplication Rule**:
   $$
   P(A \cap B) = P(A|B) P(B) = P(B|A) P(A)
   $$

3. **Chain Rule**:
   $$
   P(A_1 \cap \dots \cap A_n) = \prod_{i=1}^n P(A_i | A_1 \cap \dots \cap A_{i-1})
   $$

4. **Independence**:
   $$
   P(A \cap B) = P(A) P(B) \iff A \perp\!\!\!\perp B
   $$

5. **Law of Total Probability**:
   $$
   P(A) = \sum_{i=1}^n P(A|B_i) P(B_i)
   $$

6. **Bayes' Rule (simple form)**:
   $$
   P(A|B) = \frac{P(B|A) P(A)}{P(B)}
   $$

## Properties

1. **Normalisation**: $P(A|B) \geq 0$ and $\sum_{a} P(A=a|B) = 1$ for a fixed $B$.
2. **Range**: $0 \leq P(A|B) \leq 1$.
3. **Conditional Complement**: $P(A^c|B) = 1 - P(A|B)$.
4. **Conditional Union**: $P(A \cup C|B) = P(A|B) + P(C|B) - P(A \cap C|B)$.
5. **Independence implies $P(A|B) = P(A)$** and $P(B|A) = P(B)$.
6. **Conditional probability is a probability measure**: For fixed $B$, $P(\cdot|B)$ satisfies all Kolmogorov axioms.
7. **Bayes' rule symmetry**: $P(A|B)P(B) = P(B|A)P(A)$.

## Step-by-Step Worked Examples

### Example 1: Basic Conditional Probability with Dice

**Problem**: A fair die is rolled. Given that the result is even, what is the probability it is greater than 4?

**Solution**:

Step 1: Define events. $A = \{\text{result} > 4\} = \{5, 6\}$, $B = \{\text{even}\} = \{2, 4, 6\}$.

Step 2: Compute $P(B) = \frac{3}{6} = \frac{1}{2}$.

Step 3: Compute $P(A \cap B)$. The outcomes that are both even AND greater than 4: $\{6\}$. So $P(A \cap B) = \frac{1}{6}$.

Step 4: Apply the definition:
$$
P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{1/6}{1/2} = \frac{1}{3}
$$

Step 5: Verify intuitively. Given the die is even, the reduced sample space is $\{2, 4, 6\}$. Among these, only $\{6\}$ is greater than 4, so $P = 1/3$. Correct.

### Example 2: Multiplication Rule and Independence

**Problem**: Two cards are drawn without replacement from a 52-card deck. What is the probability both are aces?

**Solution**:

Step 1: Define events. $A_1 = \{\text{first card is ace}\}$, $A_2 = \{\text{second card is ace}\}$.

Step 2: Compute $P(A_1) = \frac{4}{52} = \frac{1}{13}$.

Step 3: Compute $P(A_2|A_1)$. Given the first card was an ace, there are 3 aces left in 51 cards. So $P(A_2|A_1) = \frac{3}{51} = \frac{1}{17}$.

Step 4: Apply the multiplication rule:
$$
P(A_1 \cap A_2) = P(A_1) P(A_2|A_1) = \frac{4}{52} \times \frac{3}{51} = \frac{12}{2652} = \frac{1}{221}
$$

Step 5: If drawing with replacement, $P(A_2|A_1) = P(A_2) = \frac{4}{52}$, and $P(A_1 \cap A_2) = \frac{1}{13} \times \frac{1}{13} = \frac{1}{169}$.

### Example 3: Law of Total Probability

**Problem**: In a certain population, 40% of people are smokers ($S$) and 60% are non-smokers ($S^c$). The probability of developing lung cancer ($C$) is 0.15 for smokers and 0.01 for non-smokers. What is the overall probability of lung cancer?

**Solution**:

Step 1: Identify the partition. $\{S, S^c\}$ partitions the population.

Step 2: Apply the law of total probability:
$$
P(C) = P(C|S) P(S) + P(C|S^c) P(S^c)
$$

Step 3: Substitute values:
$$
P(C) = 0.15 \times 0.4 + 0.01 \times 0.6 = 0.06 + 0.006 = 0.066
$$

Step 4: Interpret. The overall probability of lung cancer is 6.6%.

### Example 4: Testing Independence

**Problem**: In a survey of 200 people: 120 drink coffee ($C$), 80 drink tea ($T$), and 40 drink both. Are coffee drinking and tea drinking independent?

**Solution**:

Step 1: Compute probabilities:
- $P(C) = \frac{120}{200} = 0.6$
- $P(T) = \frac{80}{200} = 0.4$
- $P(C \cap T) = \frac{40}{200} = 0.2$

Step 2: Check independence condition $P(C \cap T) = P(C) P(T)$:
$$
P(C) P(T) = 0.6 \times 0.4 = 0.24
$$
$$
P(C \cap T) = 0.2 \neq 0.24
$$

Step 3: Since $0.2 \neq 0.24$, coffee and tea drinking are not independent. They are negatively associated (less overlap than expected under independence).

Step 4: Alternatively, $P(C|T) = \frac{0.2}{0.4} = 0.5 \neq P(C) = 0.6$, confirming dependence.

### Example 5: Chain Rule for Sequence Probability

**Problem**: A bag contains 5 red, 4 blue, and 3 green marbles. Three marbles are drawn without replacement. What is the probability the sequence is red, then blue, then green?

**Solution**:

Step 1: Define events. $R_1$ = first is red, $B_2$ = second is blue, $G_3$ = third is green.

Step 2: Compute:
- $P(R_1) = \frac{5}{5+4+3} = \frac{5}{12}$
- $P(B_2|R_1) = \frac{4}{11}$ (4 blues among 11 remaining)
- $P(G_3|R_1 \cap B_2) = \frac{3}{10}$ (3 greens among 10 remaining)

Step 3: Apply the chain rule:
$$
P(R_1 \cap B_2 \cap G_3) = \frac{5}{12} \times \frac{4}{11} \times \frac{3}{10} = \frac{60}{1320} = \frac{1}{22}
$$

Step 4: Interpret. The probability of drawing red, then blue, then green in that specific order is approximately 0.045.

## Visual Interpretation

Conditional probability can be visualised as zooming into a subset of the Venn diagram. The sample space rectangle $S$ has total area 1. Event $B$ is a region within it with area $P(B)$. The intersection $A \cap B$ is the overlap between $A$ and $B$. Conditional probability $P(A|B)$ is the area of $A \cap B$ divided by the area of $B$ — it is the proportion of $B$ that is also covered by $A$.

A tree diagram is particularly effective for conditional probability. The first branching represents the first event, with branches weighted by $P(B)$ and $P(B^c)$. Each subsequent branching represents the next event conditional on the path taken. The product of branch probabilities along a path gives the joint probability of that path.

## Common Mistakes

1. **Assuming $P(A|B) = P(B|A)$**: In general, $P(A|B) \neq P(B|A)$. For example, $P(\text{rain}|\text{clouds})$ is high, but $P(\text{clouds}|\text{rain})$ is even higher (it always rains from clouds). Confusing the two is the "prosecutor's fallacy."

2. **Confusing $P(A|B)$ with $P(A \cap B)$**: $P(A|B)$ is the probability of $A$ given $B$ has occurred. $P(A \cap B)$ is the probability that both occur. The former is typically larger because it conditions on a smaller space.

3. **Applying the multiplication rule backwards**: $P(A \cap B) = P(A|B) P(B)$ is correct. Writing $P(A \cap B) = P(A|B) P(A)$ is wrong.

4. **Ignoring the condition in the denominator**: Forgetting to divide by $P(B)$ in $P(A|B) = P(A \cap B)/P(B)$ leads to incorrectly reporting $P(A \cap B)$ instead.

5. **Assuming independence without justification**: Many real-world events are dependent. Assuming $P(A \cap B) = P(A)P(B)$ without checking is a common source of error.

6. **Misapplying the law of total probability**: The conditioning events must form a partition (mutually exclusive and exhaustive). Using overlapping events double-counts probabilities.

7. **The base rate fallacy**: Ignoring the prior probability $P(B)$ when interpreting $P(A|B)$. A positive medical test result may give high $P(\text{positive}|\text{disease})$ but low $P(\text{disease}|\text{positive})$ if the disease is rare.

## Interview Questions

### Beginner

1. **Q**: Define conditional probability $P(A|B)$.
   **A**: $P(A|B) = P(A \cap B) / P(B)$, provided $P(B) > 0$. It is the probability of $A$ occurring given that $B$ has occurred.

2. **Q**: If $P(A) = 0.5$, $P(B) = 0.4$, and $P(A \cap B) = 0.2$, compute $P(A|B)$ and $P(B|A)$.
   **A**: $P(A|B) = 0.2/0.4 = 0.5$. $P(B|A) = 0.2/0.5 = 0.4$.

3. **Q**: What does it mean for two events to be independent?
   **A**: Events $A$ and $B$ are independent if $P(A \cap B) = P(A)P(B)$, or equivalently $P(A|B) = P(A)$.

4. **Q**: State the law of total probability.
   **A**: If $\{B_1, \dots, B_n\}$ partition the sample space, then $P(A) = \sum_{i=1}^n P(A|B_i) P(B_i)$.

5. **Q**: A bag has 3 red and 7 blue marbles. One is drawn and not replaced. What is the probability the second is red given the first was red?
   **A**: After drawing one red, there are 2 red and 7 blue remaining (9 total). So $P(\text{second red} | \text{first red}) = 2/9$.

### Intermediate

1. **Q**: Prove that if $A$ and $B$ are independent, then $A$ and $B^c$ are independent.
   **A**: $P(A \cap B^c) = P(A) - P(A \cap B) = P(A) - P(A)P(B) = P(A)(1 - P(B)) = P(A)P(B^c)$. Hence $A$ and $B^c$ are independent.

2. **Q**: In machine learning, what is the conditional independence assumption made by the Naive Bayes classifier?
   **A**: Naive Bayes assumes that features are conditionally independent given the class label: $P(x_1, x_2, \dots, x_d | y) = \prod_{j=1}^d P(x_j | y)$. This simplifies computation tremendously, even though the assumption rarely holds in practice.

3. **Q**: Derive Bayes' rule from the definition of conditional probability.
   **A**: From $P(A|B) = P(A \cap B)/P(B)$ and $P(B|A) = P(A \cap B)/P(A)$, we have $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$. Solving for $P(A|B)$: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.

4. **Q**: In the context of logistic regression, what is the conditional probability being modelled?
   **A**: Logistic regression models $P(y=1|x) = \sigma(w^T x) = 1/(1+\exp(-w^T x))$, the conditional probability of the positive class given the input features. The model is trained by maximising the conditional likelihood $\prod_i P(y_i|x_i)$.

5. **Q**: What is the Markov assumption in sequence modelling?
   **A**: The Markov assumption states that the conditional probability of the current state depends only on the previous $k$ states: $P(x_t | x_{t-1}, \dots, x_1) = P(x_t | x_{t-1}, \dots, x_{t-k})$. For a first-order Markov model, $P(x_t | x_{t-1}, \dots, x_1) = P(x_t | x_{t-1})$.

### Advanced

1. **Q**: Show that conditional probability $P(\cdot|B)$ satisfies the three Kolmogorov axioms.
   **A**: (1) Non-negativity: $P(A|B) \geq 0$ because $P(A \cap B) \geq 0$ and $P(B) > 0$. (2) Normalisation: $P(S|B) = P(S \cap B)/P(B) = P(B)/P(B) = 1$. (3) Countable additivity: For disjoint $A_i$, $P(\bigcup_i A_i | B) = P((\bigcup_i A_i) \cap B)/P(B) = \sum_i P(A_i \cap B)/P(B) = \sum_i P(A_i|B)$.

2. **Q**: Explain the difference between $P(Y|do(X))$ and $P(Y|X)$ in causal inference. Provide an example.
   **A**: $P(Y|do(X))$ is the probability of $Y$ when we intervene to set $X$ to a value, while $P(Y|X)$ is the probability of $Y$ given we observe $X$ taking that value. They differ when there is confounding. Example: $P(\text{recover}|\text{take drug})$ may be high because healthier people take the drug, but $P(\text{recover}|do(\text{take drug}))$ (from a randomised trial) may show the drug is ineffective. The $do$-operator removes confounding by intervention.

3. **Q**: Derive the conditional probability chain rule for autoregressive language models and explain its connection to the product rule of probability.
   **A**: For a sequence of tokens $(w_1, w_2, \dots, w_n)$, the joint probability factorises as $P(w_1, \dots, w_n) = \prod_{t=1}^n P(w_t | w_1, \dots, w_{t-1})$ by repeated application of the multiplication rule $P(A \cap B) = P(A|B)P(B)$. Language models like GPT parameterise each conditional $P(w_t | w_{<t})$ using a neural network. The product of these conditionals gives the probability of the entire sequence, enabling generation by sampling one token at a time from the conditional distribution.

## Practice Problems

### Easy

1. $P(A) = 0.6$, $P(B) = 0.5$, $P(A \cap B) = 0.3$. Find $P(A|B)$ and $P(B|A)$.

2. A fair die is rolled. What is $P(\text{odd} | \text{less than } 5)$?

3. Two events are independent with $P(A) = 0.3$ and $P(B) = 0.4$. Find $P(A \cap B)$.

4. In a class, 70% passed math and 60% passed physics. If 50% passed both, find $P(\text{passed physics} | \text{passed math})$.

5. A card is drawn from a standard deck. What is $P(\text{heart} | \text{face card})$?

### Medium

1. A factory has two machines. Machine A produces 60% of items (1% defective), Machine B produces 40% (3% defective). What is the probability an item is defective? What is $P(\text{Machine A} | \text{defective})$?

2. Prove that if $A$ and $B$ are independent, then $P(A \cup B) = 1 - P(A^c)P(B^c)$.

3. In a population, 5% have a disease. A test is 90% sensitive (detects disease) and 85% specific (correctly identifies non-diseased). Find $P(\text{disease} | \text{positive test})$.

4. Three cards are drawn without replacement. What is the probability they are all diamonds?

5. A coin is biased: $P(H) = 0.6$. It is flipped twice. What is $P(\text{first is H} | \text{at least one H})$?

### Hard

1. The Monty Hall problem: A contestant picks one of three doors (one car, two goats). The host, who knows what is behind each door, opens a different door with a goat. The contestant may switch to the remaining door. Compute $P(\text{win} | \text{switch})$ and $P(\text{win} | \text{stay})$ using conditional probability.

2. A bag contains 2 red and 2 blue marbles. A marble is drawn. If it is red, a red marble is added; if blue, a blue marble is added. Then a second marble is drawn. What is $P(\text{second is red})$?

3. Show that $P(A \cap B | C) = P(A|B \cap C) P(B|C)$. This is a conditional version of the multiplication rule.

## Solutions

### Easy Solutions

**Solution 1**: $P(A|B) = 0.3/0.5 = 0.6$. $P(B|A) = 0.3/0.6 = 0.5$.

**Solution 2**: $A = \{1, 3, 5\}$, $B = \{1, 2, 3, 4\}$. $A \cap B = \{1, 3\}$. $P(A|B) = 2/4 = 1/2$.

**Solution 3**: $P(A \cap B) = 0.3 \times 0.4 = 0.12$.

**Solution 4**: $P(\text{physics}|\text{math}) = 0.5/0.7 = 5/7 \approx 0.714$.

**Solution 5**: $A = \text{heart}$, $B = \text{face card}$. $P(A \cap B) = 3/52$ (3 heart face cards). $P(B) = 12/52$. $P(A|B) = (3/52)/(12/52) = 3/12 = 1/4$.

### Medium Solutions

**Solution 1**: $P(D) = 0.01 \times 0.6 + 0.03 \times 0.4 = 0.006 + 0.012 = 0.018$. $P(A|D) = 0.006/0.018 = 1/3 \approx 0.333$.

**Solution 2**: $P(A \cup B) = P(A) + P(B) - P(A)P(B) = P(A) + P(B) - P(A)P(B) = 1 - (1-P(A))(1-P(B)) = 1 - P(A^c)P(B^c)$.

**Solution 3**: $P(D) = 0.05$, $P(T|D) = 0.9$, $P(T|D^c) = 0.15$. $P(D|T) = \frac{0.9 \times 0.05}{0.9 \times 0.05 + 0.15 \times 0.95} = \frac{0.045}{0.045 + 0.1425} = \frac{0.045}{0.1875} = 0.24$.

**Solution 4**: $P(\text{all diamonds}) = \frac{13}{52} \times \frac{12}{51} \times \frac{11}{50} = \frac{1716}{132600} = \frac{11}{850} \approx 0.0129$.

**Solution 5**: $P(\text{first H} \cap \text{at least one H}) = P(\text{first H}) = 0.6$. $P(\text{at least one H}) = 1 - P(\text{no H}) = 1 - 0.4^2 = 0.84$. $P(\text{first H}|\text{at least one H}) = 0.6/0.84 = 5/7 \approx 0.714$.

### Hard Solutions

**Solution 1**: Let $C$ = initial pick is car. $P(C) = 1/3$. If you initially pick the car (prob 1/3), switching loses. If you initially pick a goat (prob 2/3), the host reveals the other goat, and switching wins. So $P(\text{win}|\text{switch}) = 2/3$, $P(\text{win}|\text{stay}) = 1/3$.

**Solution 2**: $P(R_1) = 2/4 = 0.5$, $P(R_2|R_1) = 3/5$ (after adding red, 3 red, 2 blue), $P(R_2|B_1) = 2/5$ (after adding blue, 2 red, 3 blue). $P(R_2) = P(R_2|R_1)P(R_1) + P(R_2|B_1)P(B_1) = (3/5)(1/2) + (2/5)(1/2) = 3/10 + 2/10 = 0.5$.

**Solution 3**: $P(A \cap B | C) = \frac{P(A \cap B \cap C)}{P(C)} = \frac{P(A \cap B \cap C)}{P(B \cap C)} \cdot \frac{P(B \cap C)}{P(C)} = \frac{P(A \cap (B \cap C))}{P(B \cap C)} \cdot \frac{P(B \cap C)}{P(C)} = P(A | B \cap C) \cdot P(B | C)$. This is the conditional version of the multiplication rule.

## Related Concepts

- **Probability (MATH-065)**: The unconditional foundation
- **Event (MATH-067)**: The objects to which conditional probability applies
- **Bayes Theorem (MATH-069)**: Direct application of conditional probability for inference
- **Random Variable (MATH-070)**: Conditional distributions of random variables
- **Probability Distribution (MATH-071)**: Conditional distributions and joint distributions
- **Independent Events**: A special case where conditioning does not change probability

## Next Concepts

- **Bayes Theorem**: The cornerstone of Bayesian inference, derived from conditional probability (MATH-069)
- **Random Variable**: Numerical functions of outcomes with conditional distributions (MATH-070)
- **Probability Distribution**: Joint, marginal, and conditional distributions (MATH-071)

## Summary

Conditional probability $P(A|B) = P(A \cap B)/P(B)$ quantifies the probability of one event given that another has occurred. It is the foundation of the multiplication rule, the chain rule, and the law of total probability. Independence ($P(A|B) = P(A)$) is a special case where conditioning provides no information. Conditional probability is central to machine learning: classifiers output $P(y|x)$, sequence models use the Markov assumption, autoregressive models factorise joint probabilities via the chain rule, and Bayesian methods perform posterior inference. Understanding conditional probability is essential for interpreting model outputs, designing learning algorithms, and reasoning about uncertainty.

## Key Takeaways

- $P(A|B) = P(A \cap B) / P(B)$ is the definition of conditional probability
- The multiplication rule $P(A \cap B) = P(A|B)P(B)$ is used to compute joint probabilities
- Events are independent iff $P(A|B) = P(A)$ or $P(A \cap B) = P(A)P(B)$
- The law of total probability expresses $P(A)$ via a partition of the sample space
- The chain rule decomposes joint probabilities into products of conditionals
- Conditional probability is itself a probability measure satisfying Kolmogorov's axioms
- Classifiers model $P(y|x)$, the conditional probability of label given input
- The Markov assumption is a form of conditional independence
- Naive Bayes assumes conditional independence of features given the class
- Do not confuse $P(A|B)$ with $P(B|A)$ — they are generally different
