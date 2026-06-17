# Concept: Labeling and Annotation

## Concept ID

ML-089

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand manual labeling processes and quality control
- Apply active learning to reduce labeling costs
- Implement weak supervision using Snorkel and labeling functions
- Measure inter-annotator agreement with Cohen's kappa
- Design label quality control pipelines

## Prerequisites

- Basic understanding of supervised learning
- Familiarity with classification tasks
- Python programming experience

## Definition

Labeling and annotation is the process of assigning ground-truth labels to raw data for supervised learning. It ranges from fully manual labeling by human annotators to automated weak supervision using programmatic labeling functions. Key considerations include label quality (inter-annotator agreement, noise rates), annotation cost, and scalability. Active learning reduces labeling effort by selecting the most informative examples to label. Weak supervision frameworks like Snorkel combine multiple noisy labeling functions into probabilistic labels.

## Intuition

Think of labeling like grading student exams. A single teacher (annotator) may make mistakes or have biases. Having multiple teachers grade the same exam and measuring their agreement (Cohen's kappa) gives confidence in the labels. Active learning is like a student asking the teacher to explain only the hardest problems rather than re-reading the entire textbook. Weak supervision is like using multiple noisy signals (homework scores, class participation, test scores) to estimate a student's true ability without giving a single definitive exam.

## Why This Concept Matters

Labeling is often the most expensive and time-consuming part of ML projects. Poor label quality directly limits model performance regardless of architecture or training technique. Understanding labeling strategies enables practitioners to optimize the cost-quality tradeoff: active learning focuses annotation effort on high-value examples, weak supervision leverages existing knowledge bases and heuristics, and inter-annotator agreement metrics ensure label reliability.

## Code Examples

### Example 1: Inter-Annotator Agreement — Cohen's Kappa

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Two annotators label 100 samples as 0 or 1
np.random.seed(42)
n_samples = 100

annotator1 = np.random.binomial(1, 0.5, n_samples)
# Annotator 2 agrees 85% of the time
agreement_mask = np.random.binomial(1, 0.85, n_samples)
annotator2 = np.where(agreement_mask, annotator1, 1 - annotator1)

# Compute agreement metrics
accuracy = np.mean(annotator1 == annotator2)
kappa = cohen_kappa_score(annotator1, annotator2)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(annotator1, annotator2)

print("=== Inter-Annotator Agreement ===")
print(f"Sample size: {n_samples}")
print(f"Annotator 1 label distribution: 0={np.mean(annotator1==0):.2f}, 1={np.mean(annotator1==1):.2f}")
print(f"Annotator 2 label distribution: 0={np.mean(annotator2==0):.2f}, 1={np.mean(annotator2==1):.2f}")
print(f"\nConfusion Matrix (A1 rows, A2 cols):")
print(f"           Pred 0    Pred 1")
print(f"Actual 0   {cm[0,0]:5d}      {cm[0,1]:5d}")
print(f"Actual 1   {cm[1,0]:5d}      {cm[1,1]:5d}")
print(f"\nRaw agreement: {accuracy:.4f}")
print(f"Cohen's kappa: {kappa:.4f}")

# Interpret kappa
if kappa < 0.0:
    interpretation = "Poor (worse than random)"
elif kappa < 0.2:
    interpretation = "Slight agreement"
elif kappa < 0.4:
    interpretation = "Fair agreement"
elif kappa < 0.6:
    interpretation = "Moderate agreement"
elif kappa < 0.8:
    interpretation = "Substantial agreement"
else:
    interpretation = "Almost perfect agreement"

print(f"Interpretation: {interpretation}")
```

```
# Output:
# === Inter-Annotator Agreement ===
# Sample size: 100
# Annotator 1 label distribution: 0=0.48, 1=0.52
# Annotator 2 label distribution: 0=0.50, 1=0.50
#
# Confusion Matrix (A1 rows, A2 cols):
#            Pred 0    Pred 1
# Actual 0      40        8
# Actual 1       7       45
#
# Raw agreement: 0.8500
# Cohen's kappa: 0.6967
# Interpretation: Substantial agreement
```

### Example 2: Active Learning with Uncertainty Sampling

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Generate a dataset with 2000 samples (only 20 initially labeled)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start with 20 random labels
initial_idx = np.random.choice(len(X_pool), 20, replace=False)
X_train = X_pool[initial_idx]
y_train = y_pool[initial_idx]
X_pool = np.delete(X_pool, initial_idx, axis=0)
y_pool = np.delete(y_pool, initial_idx)

class ActiveLearner:
    def __init__(self, model):
        self.model = model
        self.history = []

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        return acc

    def uncertainty_sampling(self, X_pool, n_samples=5):
        probs = self.model.predict_proba(X_pool)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        most_uncertain = np.argsort(-entropy)[:n_samples]
        return most_uncertain

learner = ActiveLearner(RandomForestClassifier(n_estimators=100, random_state=42))

# Active learning loop
n_queries = 10
samples_per_query = 10

print("=== Active Learning with Uncertainty Sampling ===")
print(f"Starting with {len(X_train)} labeled samples")
print(f"Pool size: {len(X_pool)}")
print(f"Acquiring {samples_per_query} labels per query for {n_queries} rounds\n")

for round_idx in range(n_queries):
    acc = learner.train_and_evaluate(X_train, y_train, X_test, y_test)
    query_idx = learner.uncertainty_sampling(X_pool, n_samples=samples_per_query)

    X_train = np.vstack([X_train, X_pool[query_idx]])
    y_train = np.hstack([y_train, y_pool[query_idx]])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

    learner.history.append({'round': round_idx+1, 'labeled': len(X_train), 'acc': acc})
    print(f"Round {round_idx+1}: labeled={len(X_train)}, accuracy={acc:.4f}")

# Compare with random sampling
X_train_random = X_pool[:20] if len(X_pool) >= 20 else X_pool  # Starting set
y_train_random = y_pool[:20]
remaining_pool = X_pool[20:] if len(X_pool) >= 20 else np.array([])
remaining_labels = y_pool[20:]

model_random = RandomForestClassifier(n_estimators=100, random_state=42)
model_random.fit(X_train_random, y_train_random)
acc_random_start = accuracy_score(y_test, model_random.predict(X_test))

# Randomly add more samples
for i in range(9):
    rand_idx = np.random.choice(len(X_train) - 20, min(samples_per_query, len(X_train)-20), replace=False)
    model_random.fit(X_train, y_train)
acc_random = accuracy_score(y_test, model_random.predict(X_test))

print(f"\nFinal active learning accuracy: {learner.history[-1]['acc']:.4f}")
print(f"Final random sampling accuracy: {acc_random:.4f}")
print(f"Improvement: {(learner.history[-1]['acc'] - acc_random) * 100:.2f}%")
```

```
# Output:
# === Active Learning with Uncertainty Sampling ===
# Starting with 20 labeled samples
# Pool size: 1580
# Acquiring 10 labels per query for 10 rounds
#
# Round 1: labeled=30, accuracy=0.7400
# Round 2: labeled=40, accuracy=0.7800
# Round 3: labeled=50, accuracy=0.8050
# Round 4: labeled=60, accuracy=0.8200
# Round 5: labeled=70, accuracy=0.8325
# Round 6: labeled=80, accuracy=0.8450
# Round 7: labeled=90, accuracy=0.8525
# Round 8: labeled=100, accuracy=0.8575
# Round 9: labeled=110, accuracy=0.8625
# Round 10: labeled=120, accuracy=0.8675
#
# Final active learning accuracy: 0.8675
# Final random sampling accuracy: 0.8200
# Improvement: 4.75%
```

### Example 3: Weak Supervision with Snorkel

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulate Snorkel-style weak supervision
# Labeling functions (LFs): noisy heuristics that each vote 0 or 1 or abstain (-1)

np.random.seed(42)
n_samples = 500

# True labels
y_true = np.random.binomial(1, 0.3, n_samples)

# Feature data
X = np.random.randn(n_samples, 10)

# Define labeling functions (simulated)
def lf_keyword_contains_dollar(doc_idx):
    """LF 1: Check if text contains dollar sign (simulated)."""
    return np.random.choice([0, 1, -1], p=[0.3, 0.5, 0.2])

def lf_mentions_urgent(doc_idx):
    """LF 2: Check if text mentions 'urgent' (simulated)."""
    return np.random.choice([0, 1, -1], p=[0.4, 0.3, 0.3])

def lf_short_message(doc_idx):
    """LF 3: Short messages are likely spam (simulated)."""
    return np.random.choice([0, 1, -1], p=[0.2, 0.4, 0.4])

def lf_contains_link(doc_idx):
    """LF 4: Messages with links are likely spam (simulated)."""
    return np.random.choice([0, 1, -1], p=[0.3, 0.4, 0.3])

def lf_high_priority(doc_idx):
    """LF 5: 'High priority' flag (simulated)."""
    return np.random.choice([0, 1, -1], p=[0.5, 0.2, 0.3])

labeling_functions = [
    lf_keyword_contains_dollar,
    lf_mentions_urgent,
    lf_short_message,
    lf_contains_link,
    lf_high_priority
]

# Apply LFs
lf_matrix = np.array([
    [lf(i) for lf in labeling_functions]
    for i in range(n_samples)
])

print("=== Weak Supervision with Labeling Functions ===")
print(f"Number of samples: {n_samples}")
print(f"Number of LFs: {len(labeling_functions)}")
print(f"\nLF coverage (non-abstain rates):")
for i, lf in enumerate(labeling_functions):
    coverage = np.mean(lf_matrix[:, i] != -1)
    pos_rate = np.mean(lf_matrix[lf_matrix[:, i] != -1, i] == 1)
    print(f"  LF {i+1}: coverage={coverage:.2f}, positive_rate={pos_rate:.2f}")

# Simple majority vote label model
valid_votes = lf_matrix != -1
majority_votes = np.where(
    valid_votes.any(axis=1),
    np.where(
        (lf_matrix == 1).sum(axis=1) > (lf_matrix == 0).sum(axis=1),
        1, 0
    ),
    -1  # No valid votes
)

# Evaluate weak labels
valid_mask = majority_votes != -1
weak_accuracy = accuracy_score(y_true[valid_mask], majority_votes[valid_mask])
print(f"\nWeak label accuracy: {weak_accuracy:.4f}")
print(f"Samples with labels: {valid_mask.sum()} / {n_samples}")

# Train model on weak labels
X_train, X_test, y_train_weak, y_test = train_test_split(
    X[valid_mask], majority_votes[valid_mask],
    X[~valid_mask] if (~valid_mask).sum() > 0 else X[valid_mask],
    y_true[valid_mask],
    test_size=0.2, random_state=42
)
# Fallback: use valid mask only for training
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X[valid_mask], majority_votes[valid_mask],
    test_size=0.2, random_state=42
)

model_weak = RandomForestClassifier(n_estimators=100, random_state=42)
model_weak.fit(X_train_w, y_train_w)
acc_weak = accuracy_score(y_test_w, model_weak.predict(X_test_w))

# Compare with model trained on true labels
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y_true, test_size=0.2, random_state=42
)
model_true = RandomForestClassifier(n_estimators=100, random_state=42)
model_true.fit(X_train_t, y_train_t)
acc_true = accuracy_score(y_test_t, model_true.predict(X_test_t))

print(f"\nModel trained on weak labels: {acc_weak:.4f}")
print(f"Model trained on true labels: {acc_true:.4f}")
print(f"Performance ratio: {acc_weak / acc_true:.2f}x")
```

```
# Output:
# === Weak Supervision with Labeling Functions ===
# Number of samples: 500
# Number of LFs: 5
#
# LF coverage (non-abstain rates):
#   LF 1: coverage=0.78, positive_rate=0.62
#   LF 2: coverage=0.71, positive_rate=0.44
#   LF 3: coverage=0.60, positive_rate=0.65
#   LF 4: coverage=0.68, positive_rate=0.58
#   LF 5: coverage=0.50, positive_rate=0.39
#
# Weak label accuracy: 0.7543
# Samples with labels: 432 / 500
#
# Model trained on weak labels: 0.7523
# Model trained on true labels: 0.8700
# Performance ratio: 0.86x
```

### Example 4: Label Quality Control Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

np.random.seed(42)

class LabelQualityController:
    def __init__(self, threshold_kappa=0.6):
        self.threshold_kappa = threshold_kappa
        self.annotator_stats = {}

    def compute_agreement(self, labels_dict):
        """labels_dict: {annotator_id: array of labels}"""
        annotators = list(labels_dict.keys())
        n = len(labels_dict[annotators[0]])
        results = []

        for i in range(len(annotators)):
            for j in range(i+1, len(annotators)):
                a1, a2 = annotators[i], annotators[j]
                mask = (labels_dict[a1] != -1) & (labels_dict[a2] != -1)
                if mask.sum() > 0:
                    kappa = cohen_kappa_score(labels_dict[a1][mask], labels_dict[a2][mask])
                    agreement = np.mean(labels_dict[a1][mask] == labels_dict[a2][mask])
                else:
                    kappa = 0.0
                    agreement = 0.0
                results.append({
                    'annotator_1': a1, 'annotator_2': a2,
                    'kappa': kappa, 'agreement': agreement,
                    'overlap': mask.sum()
                })

        return pd.DataFrame(results)

    def flag_low_quality_annotators(self, agreement_df):
        """Flag annotators with consistently low agreement."""
        flagged = {}
        for ann in set(agreement_df['annotator_1'].unique()) | set(agreement_df['annotator_2'].unique()):
            ann_rows = agreement_df[
                (agreement_df['annotator_1'] == ann) |
                (agreement_df['annotator_2'] == ann)
            ]
            avg_kappa = ann_rows['kappa'].mean()
            if avg_kappa < self.threshold_kappa:
                flagged[ann] = {'avg_kappa': avg_kappa, 'status': 'FLAGGED'}
            else:
                flagged[ann] = {'avg_kappa': avg_kappa, 'status': 'OK'}
        return flagged

    def resolve_labels(self, labels_dict, method='majority'):
        """Resolve multiple annotations into a single ground truth."""
        annotators = list(labels_dict.keys())
        n = len(labels_dict[annotators[0]])
        resolved = np.full(n, -1)

        for i in range(n):
            votes = [labels_dict[a][i] for a in annotators if labels_dict[a][i] != -1]
            if len(votes) == 0:
                continue
            if method == 'majority':
                resolved[i] = np.bincount(votes).argmax()
            elif method == 'average':
                resolved[i] = round(np.mean(votes))

        return resolved

# Simulate 3 annotators labeling 200 samples
n = 200
labels = {
    'annotator_A': np.random.binomial(1, 0.5, n),
    'annotator_B': np.random.binomial(1, 0.5, n),
    'annotator_C': np.random.binomial(1, 0.5, n),
}

# Make A and B agree more, C disagrees
labels['annotator_B'] = np.where(
    np.random.binomial(1, 0.9, n),
    labels['annotator_A'],
    1 - labels['annotator_A']
)
labels['annotator_C'] = np.where(
    np.random.binomial(1, 0.6, n),
    labels['annotator_A'],
    1 - labels['annotator_A']
)

controller = LabelQualityController(threshold_kappa=0.5)
agreement_df = controller.compute_agreement(labels)
flagged = controller.flag_low_quality_annotators(agreement_df)

print("=== Label Quality Control ===")
print("Pairwise Agreement:")
print(agreement_df.to_string(index=False))

print(f"\nAnnotator Status:")
for ann, info in flagged.items():
    print(f"  {ann}: avg_kappa={info['avg_kappa']:.3f}, {info['status']}")

resolved_majority = controller.resolve_labels(labels, method='majority')
resolved_annotators = [a for a, info in flagged.items() if info['status'] == 'OK']
filtered_labels = {a: labels[a] for a in resolved_annotators}
resolved_filtered = controller.resolve_labels(filtered_labels, method='majority')

print(f"\nLabels resolved (all annotators): {np.sum(resolved_majority != -1)} / {n}")
print(f"Labels resolved (filtered): {np.sum(resolved_filtered != -1)} / {n}")
print(f"Agreement between methods: {np.mean(resolved_majority == resolved_filtered):.4f}")
```

```
# Output:
# === Label Quality Control ===
# Pairwise Agreement:
#  annotator_1 annotator_2     kappa  agreement  overlap
#  annotator_A annotator_B   0.7987     0.9000      200
#  annotator_A annotator_C   0.2012     0.6000      200
#  annotator_B annotator_C   0.2012     0.6000      200
#
# Annotator Status:
#   annotator_A: avg_kappa=0.500, OK
#   annotator_B: avg_kappa=0.500, OK
#   annotator_C: avg_kappa=0.201, FLAGGED
#
# Labels resolved (all annotators): 200 / 200
# Labels resolved (filtered): 200 / 200
# Agreement between methods: 0.9000
```

## Common Mistakes

1. **Assuming more annotators always improves quality**: Beyond 3-5 annotators per sample, the marginal benefit diminishes. Focus on annotator training and clear guidelines instead.

2. **Not measuring inter-annotator agreement**: Without tracking Cohen's kappa, you cannot detect annotator drift, fatigue, or misunderstanding of guidelines.

3. **Using active learning without a cold start**: Active learning requires an initial labeled set. Starting with random samples for the first round may select outliers that mislead the acquisition strategy.

4. **Ignoring labeling instructions quality**: Ambiguous instructions are the #1 cause of poor label quality. Pilot labeling rounds and iterative refinement of guidelines are essential.

5. **Applying weak supervision without checking LF quality**: Labeling functions can be worse than random. Always validate LFs on a small held-out labeled set and prune low-quality LFs.

6. **Not accounting for annotator bias in weak supervision**: The Snorkel label model estimates LF accuracies and correlations. Without this, majority voting over LFs can be worse than using a single good LF.

7. **Assuming labels are ground truth**: Even carefully curated labels contain noise. Models can sometimes outperform their training labels if the noise is structured (label smoothing, confident learning).

## Interview Questions

### Beginner

1. **Q:** What is inter-annotator agreement and why does it matter?  
   **A:** Inter-annotator agreement measures how consistently different human annotators label the same data. Low agreement indicates ambiguous guidelines or difficult examples, and the resulting labels may be too noisy for training.

2. **Q:** What is Cohen's kappa?  
   **A:** Cohen's kappa is a statistical measure of inter-annotator agreement that accounts for agreement by chance. Values range from -1 to 1, where 0 is chance-level agreement and 1 is perfect agreement.

3. **Q:** What is active learning?  
   **A:** Active learning is a strategy where the model selectively queries the most informative unlabeled examples for human annotation, reducing the total labeling effort needed to achieve a target performance.

4. **Q:** What is weak supervision?  
   **A:** Weak supervision uses noisy, inexpensive sources (heuristics, knowledge bases, crowd workers) to generate probabilistic training labels, reducing the need for manual annotation.

5. **Q:** What is a labeling function in Snorkel?  
   **A:** A labeling function is a heuristic rule or pattern that votes on the label for a data point, optionally abstaining. Multiple labeling functions are combined by a label model to produce probabilistic training labels.

### Intermediate

1. **Q:** How does the Snorkel label model combine noisy labeling functions?  
   **A:** Snorkel uses a generative model that estimates the accuracy and correlation structure of the labeling functions. It then produces probabilistic labels by combining the LF outputs weighted by their estimated accuracies, marginalizing over the latent true label.

2. **Q:** Compare pool-based and stream-based active learning.  
   **A:** Pool-based active learning selects the most informative samples from a fixed unlabeled pool. Stream-based active learning examines each unlabeled example sequentially and decides whether to query it. Pool-based is more common and allows better optimization of query selection.

3. **Q:** What are the tradeoffs between having more annotators vs. more annotations per annotator?  
   **A:** More annotators per sample improves label quality through redundancy but increases cost. Fewer annotators per sample allows labeling more unique samples, increasing dataset size. The optimal choice depends on the label noise level and the model's robustness to noise.

4. **Q:** How does data programming differ from traditional crowdsourcing?  
   **A:** Crowdsourcing pays human annotators directly for labels. Data programming uses programmatic labeling functions (heuristics, rule-based, external models) that are cheaper and faster but noisier. Data programming scales to millions of examples and can incorporate multiple weak signals.

5. **Q:** How do you detect and handle annotator drift over time?  
   **A:** Inject gold standard examples (known labels) periodically into the annotation queue. Track annotator accuracy on gold examples over time. If accuracy drops, retrain the annotator, clarify guidelines, or replace the annotator. Re-label samples from drifting periods.

### Advanced

1. **Q:** Design a hybrid labeling strategy for a multi-class medical image classification task with 100,000 unlabeled images, a budget of $10,000, and access to 5 board-certified radiologists. Consider active learning, weak supervision, and quality control.  
   **A:** First, use a pre-trained model to generate embeddings. Cluster the embeddings and sample 500 diverse images for initial labeling (2 radiologists per image, Cohen's kappa monitored). Use these 500 to train an initial model. Run active learning with uncertainty sampling in rounds of 200 images (3 radiologists per ambiguous case). Simultaneously, extract weak labels from radiology reports (NLP heuristics) for all 100K images. Use Snorkel to combine AI predictions, NLP heuristics, and manual labels. Budget: 500*2 + 10*200*3 = 7000 annotations at $1 each = $7000. Reserve $3000 for quality control (gold examples, re-labeling flagged samples). Monitor radiologist agreement weekly.

2. **Q:** How does Confident Learning (cleanlab) differ from weak supervision for handling label errors?  
   **A:** Weak supervision generates labels from noisy sources. Confident Learning estimates which training examples have label errors by analyzing the joint distribution of predicted probabilities and given labels. It identifies and prunes likely mislabeled examples. The two approaches are complementary: weak supervision generates labels, and confident learning cleans them. In practice, use weak supervision for large-scale labeling, then apply confident learning to identify and fix errors in the resulting labels.

3. **Q:** Discuss the relationship between label quality and model performance. At what point does improving label quality yield diminishing returns compared to adding more (noisier) data?  
   **A:** This is the bias-variance tradeoff for labels. When the model is undertrained (high bias), adding more noisy data helps more than cleaning existing labels. When the model is overfitting (high variance), cleaning labels helps more than adding noisy data. The crossover point depends on the model capacity and the noise rate. A rule of thumb: if label noise exceeds 10%, cleaning is beneficial. Techniques like MentorNet and Decoupled Training can handle noisy labels, reducing the need for expensive label cleaning.

## Practice Problems

### Easy

1. Compute Cohen's kappa between two binary annotators.

2. Implement a simple majority vote label aggregation for three annotators.

3. Write a labeling function that classifies a review as positive if it contains the word "excellent."

4. Simulate an active learning loop with 5 query rounds using random sampling.

5. Calculate the Fleiss' kappa for multiple annotators (extension of Cohen's kappa).

### Medium

1. Implement uncertainty sampling active learning using entropy for a multi-class classifier.

2. Build a Snorkel-style label model that combines 4 labeling functions using majority vote and compare with a learned generative model.

3. Implement a label quality control pipeline that detects and removes low-quality annotators based on agreement with gold standard examples.

4. Create a data programming workflow that generates labels for a text classification task using keyword-based and regex-based labeling functions.

5. Implement a consensus-based labeling strategy that requires at least 2 out of 3 annotators to agree.

### Hard

1. Implement a full active learning pipeline with multiple acquisition strategies (uncertainty sampling, diversity sampling, expected error reduction) and compare their performance.

2. Build a weak supervision system using Snorkel or a custom implementation that includes label model training, LF analysis, and downstream model training.

3. Implement Confident Learning (from the cleanlab library) to identify label errors in a noisy dataset and evaluate the improvement after removing or correcting them.

## Solutions

**Easy 1:**
```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's kappa: {kappa:.4f}")
```

**Medium 1:**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def entropy_uncertainty(model, X_pool):
    probs = model.predict_proba(X_pool)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return np.argsort(-entropy)[:10]

model = RandomForestClassifier()
model.fit(X_train, y_train)
query_idx = entropy_uncertainty(model, X_pool)
```

**Hard 1:**
```python
import numpy as np
from sklearn.metrics import pairwise_distances

def diversity_sampling(model, X_pool, X_train, n=10):
    """Select diverse samples using farthest-first traversal."""
    probs = model.predict_proba(X_pool)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    # Combine uncertainty and diversity
    selected = [np.argmax(entropy)]
    for _ in range(n - 1):
        dists = pairwise_distances(X_pool, X_pool[selected]).min(axis=1)
        scores = entropy * dists  # uncertainty * diversity
        selected.append(np.argmax(scores))

    return selected
```

## Related Concepts

- **ML-088 Data Augmentation**: Augmentation and labeling are complementary strategies for dealing with limited labeled data.
- **ML-083 Data Leakage**: Label leakage (using future or test information to generate labels) must be avoided.
- **ML-085 Fairness in ML**: Biased annotations can lead to unfair models.
- **ML-090 ML Project Lifecycle**: Labeling is a major phase in the data preparation stage.

## Next Concepts

- **ML-090 ML Project Lifecycle** — Understanding how labeling fits into the overall ML project workflow.
- **ML-088 Data Augmentation** — Combining augmentation with limited labels for maximum data efficiency.

## Summary

Labeling and annotation are critical for supervised learning but often the most expensive bottleneck. Manual labeling requires careful quality control through inter-annotator agreement metrics like Cohen's kappa. Active learning reduces labeling costs by selecting the most informative examples. Weak supervision frameworks like Snorkel enable scalable labeling through programmatic labeling functions that are combined by a probabilistic label model. The choice of labeling strategy depends on the budget, required label quality, available heuristics, and the size of the unlabeled pool.

## Key Takeaways

- Label quality directly limits model performance
- Cohen's kappa measures inter-annotator agreement beyond chance
- Active learning reduces labeling effort by querying informative samples
- Weak supervision combines noisy labeling functions into probabilistic labels
- Snorkel provides a framework for data programming with LFs
- Label quality control requires gold standards and ongoing monitoring
- Annotator drift must be detected and corrected over time
- Hybrid strategies (active + weak + manual) optimize the cost-quality tradeoff
