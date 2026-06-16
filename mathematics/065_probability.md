# Concept: Probability

## Concept ID

MATH-065

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- Define probability as a measure of uncertainty on a scale from 0 to 1
- State and apply the three axioms of probability (Kolmogorov axioms)
- Distinguish between classical, frequentist, and Bayesian interpretations of probability
- Compute probabilities of unions, intersections, and complements of events
- Apply the addition rule for non-mutually exclusive events
- Understand how probability underpins uncertainty quantification in AI/ML

## Prerequisites

- Basic set theory (subsets, unions, intersections, complements)
- Elementary arithmetic and fractions
- Familiarity with basic counting principles (combinations and permutations helpful but not required)

## Definition

**Probability** is a numerical measure of the likelihood that a given event will occur. It is a function $P$ that assigns to every event $A$ in a sample space $S$ a real number satisfying the three Kolmogorov axioms:

1. **Non-negativity**: $P(A) \geq 0$ for every event $A$.
2. **Normalisation**: $P(S) = 1$, where $S$ is the entire sample space.
3. **Additivity**: For any countable collection of mutually exclusive events $A_1, A_2, \dots$, we have $P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$.

From these axioms, all of probability theory follows. A probability of 0 indicates impossibility, and a probability of 1 indicates certainty. Most real-world events have probabilities strictly between 0 and 1.

## Intuition

Probability quantifies uncertainty. When you flip a fair coin, you are uncertain whether it will land heads or tails, but you know each outcome is equally likely. Probability assigns the number 0.5 to each outcome, capturing this balance of uncertainty.

Think of probability as a measure of belief or a long-run frequency. If you flip a fair coin 1,000 times, you expect roughly 500 heads. The probability of heads (0.5) reflects this long-run proportion. For one flip, the probability tells you how much confidence to place in heads versus tails — you should not be more confident in one than the other.

## Why This Concept Matters

Probability is the mathematical language of uncertainty. Every field that deals with incomplete information — science, engineering, medicine, finance, sports analytics, and especially artificial intelligence — relies on probability. Machine learning models produce probabilistic predictions (e.g., "85% chance this image contains a cat"), and understanding probability is essential for interpreting these outputs, calibrating confidence, and making decisions under uncertainty. Without probability, we cannot quantify risk, measure model uncertainty, or perform statistical inference.

## Historical Background

The modern theory of probability originated in the 17th century with correspondence between **Blaise Pascal** (1623–1662) and **Pierre de Fermat** (1607–1665) about gambling problems. They solved the "problem of points": how to divide stakes in an unfinished game of chance. **Christiaan Huygens** (1629–1695) published the first printed book on probability, *De Ratiociniis in Ludo Aleae* (On Reasoning in Games of Chance), in 1657.

**Jacob Bernoulli** (1655–1705) proved the Law of Large Numbers, and **Abraham de Moivre** (1667–1754) discovered the normal distribution. **Pierre-Simon Laplace** (1749–1827) systematised classical probability and developed Bayesian inference. The rigorous axiomatic foundation was established by **Andrey Kolmogorov** (1903–1987) in his 1933 monograph *Foundations of the Theory of Probability*, which placed probability on solid measure-theoretic footing.

## Real World Examples

1. **Weather Forecasting** — A 70% chance of rain means that, under identical atmospheric conditions, rain occurs in 7 out of 10 cases. Meteorologists use ensemble forecasting, running multiple simulations and counting how many predict rain.

2. **Medical Testing** — A COVID-19 test with 95% sensitivity correctly identifies 95% of infected individuals. The probability that you have the disease given a positive test depends on both the test accuracy and the disease prevalence (base rate).

3. **Insurance** — Actuaries compute the probability of events like car accidents, house fires, or death at various ages. These probabilities determine insurance premiums. A 25-year-old driver has a higher probability of an accident than a 45-year-old driver, so their premium is higher.

4. **Quality Control** — A factory produces microchips with a 2% defect rate. The probability that a randomly selected chip is defective is 0.02. If you buy 10 chips, probability helps compute the chance that none are defective.

5. **Sports Analytics** — A basketball player shoots free throws at 80%. The probability of making exactly 8 out of 10 free throws can be computed using the binomial distribution, helping coaches decide whether to foul a particular player.

## AI/ML Relevance

1. **Probabilistic Predictions** — Modern classifiers output probability vectors rather than hard labels. For example, a neural network trained on MNIST outputs 10 probabilities summing to 1, one for each digit class. The predicted class is typically the one with highest probability, but the entire probability vector conveys confidence.

2. **Confidence Calibration** — A well-calibrated model should have predicted probabilities that match empirical frequencies. If a model predicts 100 events with 80% probability, roughly 80 should actually occur. Miscalibrated models overestimate or underestimate uncertainty. Techniques like temperature scaling adjust model outputs to improve calibration.

3. **Uncertainty Quantification** — In safety-critical applications (autonomous driving, medical diagnosis), knowing what a model does not know is as important as knowing what it does. Probabilistic methods (Bayesian neural networks, Monte Carlo dropout) quantify epistemic uncertainty (model uncertainty) and aleatoric uncertainty (data noise).

4. **Loss Functions** — Cross-entropy loss, the most common loss for classification, is derived from probability theory. It measures the dissimilarity between the true distribution (one-hot label) and the predicted distribution (model output).

5. **Reinforcement Learning** — Policies are often probabilistic, mapping states to probability distributions over actions. The probability of taking an action determines exploration versus exploitation. Value functions estimate expected cumulative reward, which is a probability-weighted average.

6. **Generative Models** — Variational autoencoders (VAEs) and generative adversarial networks (GANs) model the probability distribution of data. VAEs maximise the Evidence Lower Bound (ELBO), a quantity rooted in probability theory.

## Mathematical Explanation

### The Three Interpretations of Probability

**Classical (Laplace)**: If there are $n$ equally likely outcomes, the probability of event $A$ is $P(A) = \frac{|A|}{n}$. This works for fair dice, coins, and cards but fails when outcomes are not equally likely.

**Frequentist**: Probability is the limit of the relative frequency as the number of trials approaches infinity: $P(A) = \lim_{n \to \infty} \frac{n_A}{n}$, where $n_A$ is the number of times $A$ occurs in $n$ trials. This requires repeatable experiments.

**Bayesian**: Probability represents a degree of belief that can be updated with new evidence. Prior beliefs $P(A)$ are combined with data likelihood $P(\text{data}|A)$ to produce posterior beliefs $P(A|\text{data})$. This is the most flexible interpretation and is widely used in machine learning.

### Basic Rules Derived from the Axioms

**Complement Rule**: $P(A^c) = 1 - P(A)$.

**Addition Rule for Mutually Exclusive Events**: If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$.

**General Addition Rule**: For any two events $A$ and $B$, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

**Monotonicity**: If $A \subseteq B$, then $P(A) \leq P(B)$.

**Bounds**: $0 \leq P(A) \leq 1$ for every event $A$.

## Formula(s)

1. **Kolmogorov Axioms**:
   $$
   P(A) \geq 0,\quad P(S) = 1,\quad P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)\ \text{for disjoint } A_i
   $$

2. **Complement Rule**:
   $$
   P(A^c) = 1 - P(A)
   $$

3. **Addition Rule (General)**:
   $$
   P(A \cup B) = P(A) + P(B) - P(A \cap B)
   $$

4. **Addition Rule (Mutually Exclusive)**:
   $$
   P(A \cup B) = P(A) + P(B)
   $$

5. **Probability of an Event in a Finite Equally Likely Sample Space**:
   $$
   P(A) = \frac{|A|}{|S|}
   $$

## Properties

1. **Non-negativity**: The probability of any event is at least 0.
2. **Upper Bound**: The probability of any event is at most 1.
3. **Certain Event**: $P(S) = 1$, the entire sample space has probability 1.
4. **Impossible Event**: $P(\emptyset) = 0$, the empty set has probability 0.
5. **Finite Additivity**: For finitely many disjoint events, $P(A_1 \cup \dots \cup A_n) = \sum_{i=1}^n P(A_i)$.
6. **Complementarity**: $P(A^c) = 1 - P(A)$.
7. **Subadditivity (Boole's Inequality)**: $P(\bigcup_i A_i) \leq \sum_i P(A_i)$.
8. **Inclusion-Exclusion**: Generalises addition rule to $n$ events, alternating sums and subtractions of intersections.

## Step-by-Step Worked Examples

### Example 1: Basic Probability with Dice

**Problem**: A fair six-sided die is rolled. What is the probability of rolling a number greater than 4?

**Solution**:

Step 1: Identify the sample space. For a fair six-sided die, $S = \{1, 2, 3, 4, 5, 6\}$, so $|S| = 6$.

Step 2: Identify the event. $A = \{\text{number} > 4\} = \{5, 6\}$, so $|A| = 2$.

Step 3: Apply the classical probability formula for equally likely outcomes:
$$
P(A) = \frac{|A|}{|S|} = \frac{2}{6} = \frac{1}{3}
$$

Step 4: Interpret. The probability of rolling a number greater than 4 on a fair die is $\frac{1}{3}$, or approximately 0.333.

### Example 2: Union of Events

**Problem**: A card is drawn from a standard 52-card deck. What is the probability that the card is either a heart or a face card (Jack, Queen, King)?

**Solution**:

Step 1: Define events. $A = \{\text{heart}\}$, $B = \{\text{face card}\}$.

Step 2: Count outcomes. There are 13 hearts and 12 face cards (3 face cards in each of 4 suits). However, 3 cards are both hearts and face cards (Jack, Queen, King of hearts).

Step 3: Apply the general addition rule:
$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

Step 4: Compute each probability:
- $P(A) = \frac{13}{52} = \frac{1}{4}$
- $P(B) = \frac{12}{52} = \frac{3}{13}$
- $P(A \cap B) = \frac{3}{52}$

Step 5: Substitute:
$$
P(A \cup B) = \frac{13}{52} + \frac{12}{52} - \frac{3}{52} = \frac{22}{52} = \frac{11}{26}
$$

Step 6: Interpret. The probability is $\frac{11}{26}$, or approximately 0.423.

### Example 3: Complement and Addition Rules

**Problem**: In a bag of 20 marbles, 8 are red, 6 are blue, 4 are green, and 2 are yellow. You draw one marble at random. (a) What is the probability it is not yellow? (b) What is the probability it is red or blue?

**Solution**:

Part (a):

Step 1: Define events. $Y = \{\text{yellow}\}$. $|Y| = 2$, $|S| = 20$.

Step 2: Compute $P(Y) = \frac{2}{20} = 0.1$.

Step 3: Apply complement rule:
$$
P(\text{not yellow}) = P(Y^c) = 1 - P(Y) = 1 - 0.1 = 0.9
$$

Part (b):

Step 1: Define events. $R = \{\text{red}\}$, $B = \{\text{blue}\}$. These are mutually exclusive (a marble cannot be both red and blue).

Step 2: Compute $P(R) = \frac{8}{20} = 0.4$, $P(B) = \frac{6}{20} = 0.3$.

Step 3: Apply addition rule for mutually exclusive events:
$$
P(R \cup B) = P(R) + P(B) = 0.4 + 0.3 = 0.7
$$

Step 4: Interpret. There is a 90% chance the marble is not yellow and a 70% chance it is red or blue.

### Example 4: Probability Using Counting (Combinations)

**Problem**: A lottery requires picking 6 numbers from 1 to 49. What is the probability of winning the jackpot with a single ticket?

**Solution**:

Step 1: The sample space consists of all possible combinations of 6 numbers from 49. Order does not matter.

Step 2: Count the total number of outcomes:
$$
|S| = \binom{49}{6} = \frac{49!}{6! \cdot 43!} = 13,983,816
$$

Step 3: The winning event $A$ contains exactly 1 combination (your chosen numbers must match all 6 drawn numbers). So $|A| = 1$.

Step 4: Compute:
$$
P(A) = \frac{1}{13,983,816} \approx 7.15 \times 10^{-8}
$$

Step 5: Interpret. The probability is approximately 1 in 14 million. This extremely small probability explains why winning the lottery is so rare.

### Example 5: Using the Axioms to Derive a Bound

**Problem**: Suppose $P(A) = 0.6$, $P(B) = 0.5$, and $P(A \cap B) = 0.2$. Find $P(A \cup B)$ and $P(A^c \cap B^c)$.

**Solution**:

Step 1: Apply the general addition rule for $P(A \cup B)$:
$$
P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.6 + 0.5 - 0.2 = 0.9
$$

Step 2: Note that $A^c \cap B^c = (A \cup B)^c$ by De Morgan's law.

Step 3: Apply the complement rule:
$$
P(A^c \cap B^c) = P((A \cup B)^c) = 1 - P(A \cup B) = 1 - 0.9 = 0.1
$$

Step 4: Verify consistency. $P(A \cup B) + P((A \cup B)^c) = 0.9 + 0.1 = 1$, confirming the result.

## Visual Interpretation

Probability can be visualised using a Venn diagram, where the sample space $S$ is represented as a rectangle (total area = 1), and events are circles inside it. The area of each circle (or region) represents its probability. Overlapping regions represent intersections.

For the general addition rule $P(A \cup B) = P(A) + P(B) - P(A \cap B)$, the Venn diagram shows why we subtract the intersection: adding the areas of circles $A$ and $B$ double-counts the overlap, so we subtract it once.

A probability line (number line from 0 to 1) is another useful visualisation. Impossible events are at 0, certain events at 1, and all other probabilities fall somewhere in between. The complement of an event is the segment from $P(A)$ to 1.

## Common Mistakes

1. **Confusing mutually exclusive with independent**: Mutually exclusive events cannot occur together ($P(A \cap B) = 0$). Independent events have $P(A \cap B) = P(A)P(B)$. Many students mistakenly treat these as the same concept.

2. **Adding probabilities without checking for overlap**: Using $P(A \cup B) = P(A) + P(B)$ when events are not mutually exclusive leads to double-counting the intersection.

3. **Assuming equally likely outcomes when they are not**: The classical formula $P(A) = |A|/|S|$ only works when all outcomes are equally likely. Medical outcomes (cure vs. no cure) are not equally likely.

4. **Misinterpreting $P(A) = 0$**: While an impossible event has probability 0, a probability of 0 does not always mean the event is impossible in continuous settings. For a continuous uniform distribution on $[0, 1]$, $P(X = 0.5) = 0$, but $X = 0.5$ is possible.

5. **Gambler's fallacy**: After several tails in a row, many believe a head is "due." Each independent coin flip still has $P(\text{heads}) = 0.5$. Past outcomes do not affect future independent trials.

6. **Ignoring the complement**: Sometimes computing $P(A^c)$ is easier than computing $P(A)$ directly. For example, $P(\text{at least one head in 10 flips})$ is easier as $1 - P(\text{no heads}) = 1 - (1/2)^{10}$.

7. **Confusing $P(A \cap B)$ with $P(A|B)$**: The probability of both $A$ and $B$ occurring is not the same as the probability of $A$ given that $B$ occurred. The latter conditions on $B$.

## Interview Questions

### Beginner

1. **Q**: What are the three Kolmogorov axioms of probability?
   **A**: (1) $P(A) \geq 0$ for every event $A$. (2) $P(S) = 1$ for the sample space $S$. (3) For disjoint events $A_1, A_2, \dots$, $P(\bigcup_i A_i) = \sum_i P(A_i)$.

2. **Q**: A fair coin is flipped three times. What is the probability of getting exactly two heads?
   **A**: There are $2^3 = 8$ equally likely outcomes. Exactly two heads occurs in 3 outcomes (HHT, HTH, THH). So $P = 3/8 = 0.375$.

3. **Q**: What is the probability of rolling a sum of 7 with two fair six-sided dice?
   **A**: There are $6 \times 6 = 36$ equally likely outcomes. Sum 7 occurs for (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) — 6 outcomes. So $P = 6/36 = 1/6$.

4. **Q**: Explain the difference between classical, frequentist, and Bayesian interpretations of probability.
   **A**: Classical: ratio of favourable to equally likely outcomes. Frequentist: limit of relative frequency over infinite trials. Bayesian: degree of belief that can be updated with evidence.

5. **Q**: If $P(A) = 0.3$, $P(B) = 0.4$, and $A$ and $B$ are mutually exclusive, what is $P(A \cup B)$?
   **A**: Since they are mutually exclusive, $P(A \cup B) = P(A) + P(B) = 0.3 + 0.4 = 0.7$.

### Intermediate

1. **Q**: Derive the general addition rule $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ from the Kolmogorov axioms.
   **A**: Write $A \cup B = (A \setminus B) \cup (A \cap B) \cup (B \setminus A)$, a union of three disjoint sets. Then $P(A \cup B) = P(A \setminus B) + P(A \cap B) + P(B \setminus A)$. Also $P(A) = P(A \setminus B) + P(A \cap B)$ and $P(B) = P(B \setminus A) + P(A \cap B)$. Substituting gives $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

2. **Q**: A bag contains 5 red and 7 blue marbles. Two marbles are drawn without replacement. What is the probability both are red?
   **A**: $P(\text{both red}) = \frac{5}{12} \cdot \frac{4}{11} = \frac{20}{132} = \frac{5}{33} \approx 0.152$.

3. **Q**: What is the difference between $P(A|B)$ and $P(A \cap B)$? Can one be larger than the other?
   **A**: $P(A \cap B)$ is the probability both occur. $P(A|B)$ is the probability of $A$ given $B$. Since $P(A|B) = P(A \cap B) / P(B)$ and $P(B) \leq 1$, we have $P(A|B) \geq P(A \cap B)$.

4. **Q**: In machine learning, why is it important that model outputs represent well-calibrated probabilities?
   **A**: A well-calibrated model's predicted probabilities match empirical frequencies. If a model predicts 80% confidence on 100 samples, about 80 should be correct. Poor calibration leads to overconfidence or underconfidence, which is dangerous in high-stakes applications like medical diagnosis or autonomous driving.

5. **Q**: How does the law of total probability relate to the probability axioms?
   **A**: If $B_1, B_2, \dots, B_n$ partition the sample space (disjoint and union equals $S$), then $P(A) = \sum_{i=1}^n P(A \cap B_i) = \sum_{i=1}^n P(A|B_i) P(B_i)$. This follows from finite additivity and the fact that $(A \cap B_i)$ are disjoint and union to $A$.

### Advanced

1. **Q**: Prove Boole's inequality: $P(\bigcup_{i=1}^n A_i) \leq \sum_{i=1}^n P(A_i)$.
   **A**: For $n=1$, equality holds. Assume true for $n-1$. Then $P(\bigcup_{i=1}^n A_i) = P((\bigcup_{i=1}^{n-1} A_i) \cup A_n) \leq P(\bigcup_{i=1}^{n-1} A_i) + P(A_n) \leq \sum_{i=1}^{n-1} P(A_i) + P(A_n) = \sum_{i=1}^n P(A_i)$, using subadditivity $P(A \cup B) \leq P(A) + P(B)$, which follows from the addition rule since $P(A \cap B) \geq 0$. By induction, the inequality holds for all $n$.

2. **Q**: Discuss the role of probability in the cross-entropy loss function used in classification.
   **A**: Cross-entropy $H(p, q) = -\sum_{x} p(x) \log q(x)$ measures the average number of bits needed to encode samples from true distribution $p$ using the model distribution $q$. In classification, $p$ is the one-hot true label and $q$ is the model's predicted probability vector. Minimising cross-entropy is equivalent to minimising the Kullback-Leibler divergence $D_{KL}(p\|q)$ and maximising the log-likelihood of the data.

3. **Q**: Explain the difference between epistemic and aleatoric uncertainty and why distinguishing them matters in AI safety.
   **A**: Aleatoric uncertainty is irreducible randomness inherent in the data (e.g., measurement noise, inherent stochasticity). Epistemic uncertainty is reducible uncertainty about the model parameters due to limited data. In AI safety, knowing which type dominates informs the appropriate response: aleatoric uncertainty suggests gathering better sensors or accepting limits, while epistemic uncertainty suggests gathering more training data or using Bayesian methods to quantify model confidence. A system that confuses the two may be dangerously overconfident.

## Practice Problems

### Easy

1. A fair coin is flipped. What is the probability of getting heads?

2. A standard six-sided die is rolled. What is the probability of rolling an even number?

3. From a standard 52-card deck, what is the probability of drawing an ace?

4. A bag contains 3 red, 5 green, and 2 yellow marbles. One marble is drawn at random. What is the probability it is green?

5. In a class of 30 students, 18 are female. One student is selected at random. What is the probability the student is male?

### Medium

1. Two fair dice are rolled. What is the probability that the sum is at least 10?

2. In a group of 100 people, 40 own a cat, 30 own a dog, and 15 own both. What is the probability that a randomly selected person owns a cat or a dog?

3. A fair coin is flipped 4 times. What is the probability of getting at least one tail?

4. A password consists of 4 digits (0-9). What is the probability that a randomly generated password contains no repeated digits?

5. Three cards are drawn without replacement from a standard deck. What is the probability that all three are hearts?

### Hard

1. The birthday problem: In a group of $n$ people, what is the probability that at least two share the same birthday? Compute for $n = 23$ using the complement rule.

2. A bag contains 6 red and 4 blue marbles. Four marbles are drawn without replacement. What is the probability that exactly 2 are red?

3. Prove that $P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$ using the axioms of probability.

## Solutions

### Easy Solutions

**Solution 1**: $P(\text{heads}) = \frac{1}{2} = 0.5$.

**Solution 2**: Even numbers are $\{2, 4, 6\}$, so $P = \frac{3}{6} = \frac{1}{2}$.

**Solution 3**: There are 4 aces, so $P = \frac{4}{52} = \frac{1}{13} \approx 0.077$.

**Solution 4**: $P(\text{green}) = \frac{5}{3+5+2} = \frac{5}{10} = \frac{1}{2}$.

**Solution 5**: 12 males out of 30, so $P = \frac{12}{30} = \frac{2}{5} = 0.4$.

### Medium Solutions

**Solution 1**: Sum at least 10 means sums 10, 11, 12. Outcomes: (4,6), (5,5), (6,4), (5,6), (6,5), (6,6) — 6 outcomes. $P = 6/36 = 1/6$.

**Solution 2**: $P(\text{cat} \cup \text{dog}) = \frac{40}{100} + \frac{30}{100} - \frac{15}{100} = \frac{55}{100} = 0.55$.

**Solution 3**: Complement: $P(\text{at least one tail}) = 1 - P(\text{no tails}) = 1 - (1/2)^4 = 1 - 1/16 = 15/16 = 0.9375$.

**Solution 4**: Total passwords: $10^4 = 10,000$. Without repeats: $10 \times 9 \times 8 \times 7 = 5040$. $P = 5040/10000 = 0.504$.

**Solution 5**: $P = \frac{13}{52} \cdot \frac{12}{51} \cdot \frac{11}{50} = \frac{1716}{132600} = \frac{11}{850} \approx 0.0129$.

### Hard Solutions

**Solution 1**: $P(\text{at least one shared birthday}) = 1 - \frac{365}{365} \cdot \frac{364}{365} \cdot \dots \cdot \frac{365-n+1}{365}$. For $n = 23$: $P \approx 0.507$. So with just 23 people, there is over a 50% chance of a shared birthday.

**Solution 2**: Total outcomes: $\binom{10}{4} = 210$. Favourable (choose 2 red from 6 and 2 blue from 4): $\binom{6}{2} \cdot \binom{4}{2} = 15 \cdot 6 = 90$. $P = 90/210 = 3/7 \approx 0.429$.

**Solution 3**: Start with $P(A \cup B \cup C) = P((A \cup B) \cup C) = P(A \cup B) + P(C) - P((A \cup B) \cap C)$. Expand $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. Expand $P((A \cup B) \cap C) = P((A \cap C) \cup (B \cap C)) = P(A \cap C) + P(B \cap C) - P(A \cap B \cap C)$. Substitute: $P(A \cup B \cup C) = P(A) + P(B) - P(A \cap B) + P(C) - [P(A \cap C) + P(B \cap C) - P(A \cap B \cap C)] = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$.

## Related Concepts

- **Sample Space (MATH-066)**: The set of all possible outcomes over which probability is defined
- **Event (MATH-067)**: A subset of the sample space to which probability is assigned
- **Conditional Probability (MATH-068)**: Probability of an event given that another has occurred
- **Bayes Theorem (MATH-069)**: Updates probabilities based on new evidence
- **Random Variable (MATH-070)**: A function mapping outcomes to real numbers
- **Probability Distribution (MATH-071)**: Complete description of a random variable's probabilities
- **Set Theory**: The foundational language for defining events and sample spaces

## Next Concepts

- **Conditional Probability**: How to update probabilities when partial information is available (MATH-068)
- **Bayes Theorem**: The mathematical framework for learning from data (MATH-069)
- **Random Variable**: Assigning numerical values to outcomes (MATH-070)
- **Probability Distribution**: Characterising the behaviour of random variables (MATH-071)

## Summary

Probability is the mathematical framework for quantifying uncertainty. Governed by the three Kolmogorov axioms — non-negativity, normalisation, and additivity — probability assigns numbers between 0 and 1 to events. The classical interpretation assumes equally likely outcomes, the frequentist interpretation relies on long-run frequencies, and the Bayesian interpretation treats probability as a degree of belief. Key rules include the complement rule, the addition rule (with correction for overlap), and the inclusion-exclusion principle for multiple events. Probability is foundational to machine learning, appearing in loss functions, confidence calibration, uncertainty quantification, and generative modelling.

## Key Takeaways

- Probability is a number between 0 and 1 measuring the likelihood of an event
- The three Kolmogorov axioms are the foundation of all probability theory
- The complement rule $P(A^c) = 1 - P(A)$ is one of the most useful computational tools
- The general addition rule $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ avoids double-counting
- Classical, frequentist, and Bayesian interpretations serve different purposes
- Mutual exclusivity ($A \cap B = \emptyset$) simplifies the addition rule
- Machine learning relies on probability for predictions, calibration, and uncertainty quantification
- Cross-entropy loss and log-likelihood are directly derived from probability theory
- Understanding the axioms enables derivation of all other probability results
- Always verify that probabilities sum to 1 as a sanity check
