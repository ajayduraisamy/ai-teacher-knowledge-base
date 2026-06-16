# Concept: Event

## Concept ID

MATH-067

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Probability

## Learning Objectives

- Define an event as a subset of the sample space
- Distinguish between simple and compound events
- Define and identify mutually exclusive and exhaustive events
- Compute complements, unions, and intersections of events
- Apply event operations to probability calculations
- Connect event concepts to classification outcomes in machine learning

## Prerequisites

- Set theory (subsets, unions, intersections, complements)
- Sample space concept (MATH-066)
- Basic probability axioms (MATH-065)

## Definition

An **event** is a subset of the sample space $S$. If the outcome of a random experiment belongs to a given event $A$, we say that event $A$ has occurred. Events are denoted by capital letters $A, B, C, \dots$ and probabilities are assigned to events via a probability measure $P$.

Formally, if $S$ is the sample space, then any subset $A \subseteq S$ is an event. The empty set $\emptyset$ is called the impossible event, and $S$ itself is called the certain event.

## Intuition

An event is a collection of outcomes that share a property of interest. When rolling a die, "rolling an even number" is the event $A = \{2, 4, 6\}$. If the die shows 4, then $A$ occurs. If it shows 5, $A$ does not occur.

Think of an event as a question you ask about the outcome. "Did we get an even number?" "Did it rain today?" "Is this email spam?" The event is the set of outcomes for which the answer is yes.

## Why This Concept Matters

Events are the basic units of probability. Every probability statement involves an event: $P(A)$ is the probability that event $A$ occurs. Understanding how to combine events using complements, unions, and intersections is essential for computing probabilities of complex scenarios. In machine learning, events correspond to classification outcomes (true positive, false positive), threshold exceedances, and decision regions. The language of events translates directly to model evaluation metrics.

## Historical Background

The treatment of events as subsets of a sample space was formalised by **Andrey Kolmogorov** (1933) in his axiomatisation of probability. Before Kolmogorov, events were discussed informally. Kolmogorov's key insight was that events form a sigma-algebra — a collection of subsets closed under complementation and countable unions — ensuring that probability can be defined consistently. This set-theoretic view unified probability with measure theory and enabled the rigorous treatment of continuous sample spaces.

## Real World Examples

1. **Medical Testing** — The event "patient has disease" is $A = \{\text{disease present}\}$. The event "test is positive" is $B = \{\text{positive result}\}$. Their intersection $A \cap B$ is a true positive. Their complements correspond to false negatives, false positives, and true negatives.

2. **Weather Forecasts** — The event "precipitation exceeds 1 inch" defines a flood warning threshold. Meteorologists compute $P(\text{precipitation} > 1\text{ inch})$ using historical data.

3. **Quality Control** — The event "defective item" is monitored in manufacturing. The complement "non-defective item" is the acceptable outcome. The intersection of "defective" and "from production line 2" helps identify problem lines.

4. **Sports Betting** — The event "Team A wins by more than 7 points" defines a point spread bet. Bookmakers set odds based on $P(\text{Team A wins by } > 7)$.

5. **Cybersecurity** — The event "network intrusion detected" triggers an alert. Security analysts care about the intersection of "intrusion occurred" and "alert triggered" (true positive) versus "no intrusion" and "alert triggered" (false positive).

## AI/ML Relevance

1. **Binary Classification Events** — In binary classification, four fundamental events arise from the confusion matrix:
   - True Positive (TP): predicted positive and actually positive
   - True Negative (TN): predicted negative and actually negative
   - False Positive (FP): predicted positive but actually negative (Type I error)
   - False Negative (FN): predicted negative but actually positive (Type II error)
   
   These events define all standard metrics: precision = TP/(TP+FP), recall = TP/(TP+FN), accuracy = (TP+TN)/(TP+TN+FP+FN), and F1-score.

2. **Confidence Threshold Events** — A classifier's confidence threshold $\tau$ defines the event $A_\tau = \{x \mid P(\text{positive} \mid x) \geq \tau\}$. Varying $\tau$ trades off precision and recall, generating the Receiver Operating Characteristic (ROC) curve. Each threshold defines a different event.

3. **Multi-Class Events** — In multi-class classification with $K$ classes, each class corresponds to an event $C_k = \{\text{outcome is class } k\}$. The events partition the sample space: they are mutually exclusive and exhaustive.

4. **Rare Event Detection** — Fraud detection, anomaly detection, and failure prediction focus on rare events where $P(A)$ is very small (e.g., $10^{-6}$). Models must be designed to detect these rare events without overwhelming false positives.

5. **Event Spaces in Reinforcement Learning** — In RL, reaching a terminal state, collecting a reward, or exceeding a cost threshold are all events. Policy optimisation involves maximising the probability of favourable events (rewards) while avoiding unfavourable ones (costs, failures).

## Mathematical Explanation

### Types of Events

**Simple Event**: Contains exactly one outcome. For a die roll, $A = \{3\}$ is a simple event.

**Compound Event**: Contains two or more outcomes. $A = \{2, 4, 6\}$ (even numbers) is a compound event.

**Certain Event**: The entire sample space $S$. It always occurs.

**Impossible Event**: The empty set $\emptyset$. It never occurs.

### Event Operations

Given events $A, B \subseteq S$:

- **Complement**: $A^c = S \setminus A = \{\text{outcomes not in } A\}$. Occurs when $A$ does not occur.
- **Union**: $A \cup B = \{\text{outcomes in } A \text{ or } B \text{ or both}\}$. Occurs if at least one of $A, B$ occurs.
- **Intersection**: $A \cap B = \{\text{outcomes in both } A \text{ and } B\}$. Occurs if both $A$ and $B$ occur.

### Special Relationships

- **Mutually Exclusive (Disjoint)**: $A \cap B = \emptyset$. Events cannot occur simultaneously.
- **Exhaustive**: $A \cup B = S$. At least one of the events must occur.
- **Partition**: A collection of events that are both mutually exclusive and exhaustive.

### De Morgan's Laws

$$
(A \cup B)^c = A^c \cap B^c
$$
$$
(A \cap B)^c = A^c \cup B^c
$$

These laws are essential for simplifying event expressions.

## Formula(s)

1. **Complement Probability**:
   $$
   P(A^c) = 1 - P(A)
   $$

2. **Union Probability (General)**:
   $$
   P(A \cup B) = P(A) + P(B) - P(A \cap B)
   $$

3. **Union Probability (Mutually Exclusive)**:
   $$
   P(A \cup B) = P(A) + P(B)
   $$

4. **Intersection Probability (Independent Events)**:
   $$
   P(A \cap B) = P(A) P(B)
   $$

5. **De Morgan's Laws**:
   $$
   (A \cup B)^c = A^c \cap B^c,\quad (A \cap B)^c = A^c \cup B^c
   $$

## Properties

1. **Subset Relation**: If $A \subseteq B$, then $P(A) \leq P(B)$.
2. **Monotonicity**: $P(A \cap B) \leq P(A) \leq P(A \cup B)$.
3. **Intersection Bound**: $P(A \cap B) \leq \min(P(A), P(B))$.
4. **Union Bound**: $P(A \cup B) \geq \max(P(A), P(B))$.
5. **Complementation**: $P(A^c) = 1 - P(A)$.
6. **Null Event**: $P(\emptyset) = 0$.
7. **Sigma-Algebra**: For technical reasons, the collection of events must be closed under complementation and countable unions (a sigma-algebra), ensuring that all event combinations are valid probability assignments.

## Step-by-Step Worked Examples

### Example 1: Identifying Events from a Die Roll

**Problem**: A fair six-sided die is rolled. Let $A$ be the event of rolling an odd number, $B$ be the event of rolling a number greater than 3. (a) List the outcomes in $A$, $B$, $A \cap B$, and $A \cup B$. (b) Compute $P(A^c)$.

**Solution**:

Step 1: Write the sample space: $S = \{1, 2, 3, 4, 5, 6\}$.

Step 2: Identify events:
- $A = \{1, 3, 5\}$ (odd numbers)
- $B = \{4, 5, 6\}$ (greater than 3)

Step 3: Compute $A \cap B = \{5\}$ (odd AND greater than 3).

Step 4: Compute $A \cup B = \{1, 3, 4, 5, 6\}$ (odd OR greater than 3).

Step 5: Compute $P(A^c) = P(\{2, 4, 6\}) = \frac{3}{6} = \frac{1}{2}$.

### Example 2: Mutually Exclusive Events

**Problem**: In a deck of 52 cards, let $A$ be the event of drawing a heart, $B$ be the event of drawing a club. Are $A$ and $B$ mutually exclusive? Compute $P(A \cup B)$.

**Solution**:

Step 1: Check mutual exclusivity. $A = \{\text{all hearts}\}$, $B = \{\text{all clubs}\}$. A card cannot be both a heart and a club, so $A \cap B = \emptyset$. Yes, they are mutually exclusive.

Step 2: Compute $P(A) = \frac{13}{52} = \frac{1}{4}$, $P(B) = \frac{13}{52} = \frac{1}{4}$.

Step 3: Apply addition rule for mutually exclusive events:
$$
P(A \cup B) = P(A) + P(B) = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}
$$

Step 4: Interpret. There is a 50% chance of drawing either a heart or a club.

### Example 3: Complements and De Morgan's Laws

**Problem**: In a survey, 60% of people like coffee ($C$), 50% like tea ($T$), and 30% like both. (a) Find $P(C \cup T)$. (b) Find $P(C^c \cap T^c)$ using De Morgan's law.

**Solution**:

Part (a):

Step 1: Apply the general addition rule:
$$
P(C \cup T) = P(C) + P(T) - P(C \cap T) = 0.6 + 0.5 - 0.3 = 0.8
$$

Part (b):

Step 1: Note $C^c \cap T^c = (C \cup T)^c$ by De Morgan's law.

Step 2: Compute the complement:
$$
P(C^c \cap T^c) = P((C \cup T)^c) = 1 - P(C \cup T) = 1 - 0.8 = 0.2
$$

Step 3: Interpret. 20% of people like neither coffee nor tea.

### Example 4: Events in Binary Classification

**Problem**: A spam classifier processes 1,000 emails. The confusion matrix shows: TP = 150, TN = 750, FP = 50, FN = 50. Define the events and compute precision and recall.

**Solution**:

Step 1: Define events:
- $P$ (actual positive): email is spam
- $N$ (actual negative): email is not spam
- $P_{\text{pred}}$ (predicted positive): classifier predicts spam
- $N_{\text{pred}}$ (predicted negative): classifier predicts not spam

Step 2: Map confusion matrix entries:
- TP = $P(P_{\text{pred}} \cap P) = 150$
- TN = $P(N_{\text{pred}} \cap N) = 750$
- FP = $P(P_{\text{pred}} \cap N) = 50$
- FN = $P(N_{\text{pred}} \cap P) = 50$

Step 3: Compute precision:
$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{150}{150 + 50} = \frac{150}{200} = 0.75
$$

Step 4: Compute recall:
$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{150}{150 + 50} = \frac{150}{200} = 0.75
$$

Step 5: Interpret. The classifier is 75% precise (75% of predicted spam is actually spam) and has 75% recall (it catches 75% of actual spam).

### Example 5: Partition and Total Probability

**Problem**: A factory has two machines. Machine 1 produces 60% of items with a 2% defect rate. Machine 2 produces 40% of items with a 5% defect rate. Define events and compute the overall defect probability.

**Solution**:

Step 1: Define events:
- $M_1$: item came from Machine 1
- $M_2$: item came from Machine 2
- $D$: item is defective

Step 2: Note $M_1$ and $M_2$ partition the sample space (every item is from exactly one machine).

Step 3: Given probabilities:
- $P(M_1) = 0.6$, $P(D|M_1) = 0.02$
- $P(M_2) = 0.4$, $P(D|M_2) = 0.05$

Step 4: Apply the law of total probability:
$$
P(D) = P(D|M_1)P(M_1) + P(D|M_2)P(M_2) = 0.02 \times 0.6 + 0.05 \times 0.4 = 0.012 + 0.02 = 0.032
$$

Step 5: Interpret. The overall defect rate is 3.2%.

## Visual Interpretation

Events can be visualised using Venn diagrams, where the sample space $S$ is a rectangle and events are circles (or ovals) within it. The area of each region represents its probability.

- The complement $A^c$ is everything outside circle $A$.
- The union $A \cup B$ is the total area covered by both circles.
- The intersection $A \cap B$ is the overlapping region.
- Mutually exclusive events are drawn as non-overlapping circles.

A tree diagram is another useful visualisation for sequential events. Each branch represents the occurrence or non-occurrence of an event at a given stage, and paths through the tree represent compound events.

## Common Mistakes

1. **Confusing events with outcomes**: An event is a set of outcomes, not a single outcome (unless it is a simple event). Saying "the event is rolling a 5" is acceptable shorthand, but technically the event is the set $\{5\}$.

2. **Forgetting that events are subsets**: Every event must be a subset of the sample space. An event like "the die shows 7" on a six-sided die is $\emptyset$, the impossible event, because 7 is not in $S$.

3. **Misapplying De Morgan's laws**: A common error is writing $(A \cup B)^c = A^c \cup B^c$ instead of $(A \cup B)^c = A^c \cap B^c$. The complement of a union is the intersection of the complements.

4. **Assuming events are mutually exclusive when they are not**: Adding probabilities without subtracting the intersection leads to double-counting. Always check whether $A \cap B = \emptyset$ before using the simple addition rule.

5. **Confusing independent and mutually exclusive**: Independent events can occur together and have $P(A \cap B) = P(A)P(B)$. Mutually exclusive events cannot occur together and have $P(A \cap B) = 0$. These are fundamentally different concepts.

6. **Misidentifying complements**: The complement of $A$ is everything in $S$ not in $A$, not just the "opposite" in a colloquial sense. For example, if $A = \{2, 4, 6\}$ on a die, $A^c = \{1, 3, 5\}$, not "odd numbers plus something else."

7. **Ignoring the sigma-algebra requirement**: In continuous sample spaces, not every subset is an event. Only measurable sets (those in the Borel sigma-algebra) qualify. This technicality ensures that paradoxical sets (like the Banach-Tarski paradox) are excluded.

## Interview Questions

### Beginner

1. **Q**: What is the difference between a simple event and a compound event?
   **A**: A simple event contains exactly one outcome (e.g., $\{3\}$ on a die). A compound event contains two or more outcomes (e.g., $\{2, 4, 6\}$ for even numbers).

2. **Q**: If $P(A) = 0.4$ and $P(B) = 0.3$ and $A$ and $B$ are mutually exclusive, what is $P(A \cup B)$?
   **A**: $P(A \cup B) = P(A) + P(B) = 0.4 + 0.3 = 0.7$ because they are disjoint.

3. **Q**: What does it mean for events $A$ and $B$ to be exhaustive?
   **A**: Exhaustive means $A \cup B = S$. Every outcome in the sample space belongs to at least one of the events.

4. **Q**: State De Morgan's law for $(A \cap B)^c$.
   **A**: $(A \cap B)^c = A^c \cup B^c$.

5. **Q**: In a standard deck of cards, let $A$ be drawing a king and $B$ be drawing a queen. Are these events mutually exclusive?
   **A**: Yes, they are mutually exclusive because a single card cannot be both a king and a queen.

### Intermediate

1. **Q**: In machine learning classification, what are the four events defined by the confusion matrix?
   **A**: True Positive (predicted positive, actually positive), True Negative (predicted negative, actually negative), False Positive (predicted positive, actually negative), False Negative (predicted negative, actually positive).

2. **Q**: If $P(A) = 0.5$, $P(B) = 0.4$, and $P(A \cap B) = 0.1$, find $P(A^c \cup B^c)$.
   **A**: By De Morgan's law, $A^c \cup B^c = (A \cap B)^c$. So $P(A^c \cup B^c) = 1 - P(A \cap B) = 1 - 0.1 = 0.9$.

3. **Q**: Explain how varying a confidence threshold changes the event definition in a classifier.
   **A**: The event "positive prediction" is defined as $\{x \mid P(y=1|x) \geq \tau\}$. Lowering $\tau$ makes this event larger (more positive predictions, increasing recall but potentially decreasing precision). Raising $\tau$ shrinks the event, increasing precision at the cost of recall.

4. **Q**: Can two events be both mutually exclusive and independent?
   **A**: If $A$ and $B$ are mutually exclusive with nonzero probabilities, then $P(A \cap B) = 0$ but $P(A)P(B) > 0$, so they cannot be independent. The only way to be both is if at least one has probability 0.

5. **Q**: What is a partition of the sample space and why is it useful?
   **A**: A partition is a collection of events that are mutually exclusive and exhaustive. Partitions enable the law of total probability, which expresses $P(A)$ as a weighted average of conditional probabilities $P(A|B_i)$.

### Advanced

1. **Q**: Prove that if $A$ and $B$ are independent, then $A$ and $B^c$ are also independent.
   **A**: $P(A \cap B^c) = P(A) - P(A \cap B) = P(A) - P(A)P(B) = P(A)(1 - P(B)) = P(A)P(B^c)$. Therefore $A$ and $B^c$ are independent.

2. **Q**: In the context of continuous sample spaces, why must the collection of events form a sigma-algebra?
   **A**: A sigma-algebra ensures closure under complementation and countable unions, which is necessary for probability measures to satisfy the axioms. Without it, we might need to assign probabilities to pathological non-measurable sets, which leads to contradictions (e.g., Banach-Tarski paradox). The Borel sigma-algebra on $\mathbb{R}$ is the smallest sigma-algebra containing all intervals.

3. **Q**: Derive the inclusion-exclusion principle for three events $P(A \cup B \cup C)$.
   **A**: $P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$. This is proved by applying the two-event formula twice: $P((A \cup B) \cup C) = P(A \cup B) + P(C) - P((A \cup B) \cap C)$, then expanding $P(A \cup B)$ and $P((A \cap C) \cup (B \cap C))$.

## Practice Problems

### Easy

1. A die is rolled. Let $A = \{1, 3, 5\}$. What is $A^c$?

2. Two events $A$ and $B$ have $P(A) = 0.3$, $P(B) = 0.4$, and $P(A \cap B) = 0.1$. Compute $P(A \cup B)$.

3. A card is drawn from a 52-card deck. Let $A$ be drawing a spade and $B$ be drawing a face card. List 3 outcomes in $A \cap B$.

4. Are the events "rolling a 1" and "rolling a 6" on a single die roll mutually exclusive?

5. If $P(A) = 0.7$, what is $P(A^c)$?

### Medium

1. In a group of students, 60% study math ($M$), 50% study physics ($P$), and 30% study both. Find $P(M \cup P)$ and $P(M^c \cap P^c)$.

2. Prove using De Morgan's laws that $P(A^c \cup B^c) = 1 - P(A \cap B)$.

3. A fair coin is flipped 3 times. Let $A$ be the event "at least two heads" and $B$ be the event "first flip is heads." List the outcomes in $A \cap B$.

4. Events $A$ and $B$ are independent with $P(A) = 0.2$ and $P(B) = 0.3$. Compute $P(A \cap B)$ and $P(A \cup B)$.

5. A spam filter has 95% recall and 99% precision on a stream where 2% of emails are spam. What is the probability that a flagged email is actually spam?

### Hard

1. Three events satisfy: $P(A) = 0.5$, $P(B) = 0.4$, $P(C) = 0.3$, $P(A \cap B) = 0.2$, $P(A \cap C) = 0.15$, $P(B \cap C) = 0.1$, $P(A \cap B \cap C) = 0.05$. Compute $P(A \cup B \cup C)$.

2. Show that for any two events $A$ and $B$, $P(A \cap B) \geq P(A) + P(B) - 1$ (Bonferroni inequality).

3. A medical test for a disease has sensitivity 95% (TP rate) and specificity 98% (TN rate). The disease prevalence is 1%. Define the events and compute the probability that a randomly selected person who tests positive actually has the disease. Interpret the result.

## Solutions

### Easy Solutions

**Solution 1**: $A^c = \{2, 4, 6\}$, the even numbers.

**Solution 2**: $P(A \cup B) = 0.3 + 0.4 - 0.1 = 0.6$.

**Solution 3**: $A \cap B$ is the set of spade face cards: Jack of Spades, Queen of Spades, King of Spades.

**Solution 4**: Yes, because a single die roll cannot show both 1 and 6.

**Solution 5**: $P(A^c) = 1 - 0.7 = 0.3$.

### Medium Solutions

**Solution 1**: $P(M \cup P) = 0.6 + 0.5 - 0.3 = 0.8$. $P(M^c \cap P^c) = P((M \cup P)^c) = 1 - 0.8 = 0.2$.

**Solution 2**: By De Morgan's law, $A^c \cup B^c = (A \cap B)^c$. Therefore $P(A^c \cup B^c) = P((A \cap B)^c) = 1 - P(A \cap B)$.

**Solution 3**: $S = \{HHH, HHT, HTH, THH, HTT, THT, TTH, TTT\}$. $A = \{HHH, HHT, HTH, THH\}$, $B = \{HHH, HHT, HTH, HTT\}$. $A \cap B = \{HHH, HHT, HTH\}$.

**Solution 4**: $P(A \cap B) = 0.2 \times 0.3 = 0.06$ (independence). $P(A \cup B) = 0.2 + 0.3 - 0.06 = 0.44$.

**Solution 5**: Let $S$ be spam, $F$ be flagged. Given $P(S) = 0.02$, $P(F|S) = 0.95$, $P(F|S^c) = 1 - 0.99 = 0.01$. $P(S|F) = \frac{0.95 \times 0.02}{0.95 \times 0.02 + 0.01 \times 0.98} = \frac{0.019}{0.019 + 0.0098} = \frac{0.019}{0.0288} \approx 0.66$. So 66% of flagged emails are actually spam.

### Hard Solutions

**Solution 1**: $P(A \cup B \cup C) = 0.5 + 0.4 + 0.3 - 0.2 - 0.15 - 0.1 + 0.05 = 0.8$.

**Solution 2**: From the addition rule, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. Since $P(A \cup B) \leq 1$, we have $P(A) + P(B) - P(A \cap B) \leq 1$, which rearranges to $P(A \cap B) \geq P(A) + P(B) - 1$.

**Solution 3**: Let $D$ = has disease, $T$ = tests positive. $P(D) = 0.01$, $P(T|D) = 0.95$, $P(T|D^c) = 0.02$. $P(D|T) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.02 \times 0.99} = \frac{0.0095}{0.0095 + 0.0198} = \frac{0.0095}{0.0293} \approx 0.324$. Despite the test being accurate, only about 32% of positive tests are actually diseased due to the low prevalence. This illustrates the base rate fallacy.

## Related Concepts

- **Probability (MATH-065)**: The measure assigned to events
- **Sample Space (MATH-066)**: The set containing all outcomes from which events are formed
- **Conditional Probability (MATH-068)**: Probability of one event given another
- **Bayes Theorem (MATH-069)**: Updating event probabilities given evidence
- **Random Variable (MATH-070)**: A function that maps events to real numbers
- **Set Theory**: The mathematical foundation for event operations

## Next Concepts

- **Conditional Probability**: How the occurrence of one event affects the probability of another (MATH-068)
- **Bayes Theorem**: The fundamental rule for updating beliefs based on evidence (MATH-069)

## Summary

An event is a subset of the sample space representing a collection of outcomes of interest. Simple events contain one outcome; compound events contain multiple. Events can be combined using complementation, union, and intersection, with De Morgan's laws relating these operations. Mutually exclusive events cannot occur together, while exhaustive events cover the entire sample space. Event concepts translate directly to machine learning, where confusion matrix entries (TP, TN, FP, FN) are intersection events, and confidence thresholds define decision events. Mastery of event operations is essential for probability computation and model evaluation.

## Key Takeaways

- An event is a subset of the sample space $S$
- $A^c$ (complement), $A \cup B$ (union), and $A \cap B$ (intersection) are the basic event operations
- Mutually exclusive events have $A \cap B = \emptyset$
- De Morgan's laws relate complements of unions and intersections
- The four confusion matrix events (TP, TN, FP, FN) are intersection events
- Confidence thresholds define decision regions as events
- A partition is a set of events that are mutually exclusive and exhaustive
- The law of total probability uses partitions to compute unconditional probabilities
- Event notation is the language for communicating probability statements
- Always verify that events are valid subsets of the specified sample space
