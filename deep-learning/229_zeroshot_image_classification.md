# Concept: Zero-shot Image Classification

## Concept ID

DL-229

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand zero-shot learning and its assumptions
- Implement zero-shot classification using CLIP-style models
- Analyze the role of semantic embeddings
- Evaluate zero-shot classification performance

## Prerequisites

DL-228 Few-shot Image Classification, DL-224 Feature Extraction

## Definition

Zero-shot image classification is the ability to classify images into classes that were never seen during training, by leveraging semantic side information (e.g., textual descriptions, attribute vectors) that bridges seen and unseen classes.

## Intuition

If you've never seen a zebra but you know it's "a striped horse-like animal," you can recognize one. Zero-shot learning works similarly: the model learns a shared embedding space where images and class descriptions are mapped to the same representation. During training, the model sees images of "horses" and textual descriptions like "a large mammal used for riding." At test time, it sees a zebra and the description "a striped horse-like African animal" — even though it never trained on zebra images, the alignment between the visual features and the textual description allows correct classification.

## Why This Concept Matters

Zero-shot learning dramatically expands the applicability of image classification to open-set scenarios where new classes appear constantly. Models like CLIP have shown remarkable zero-shot capabilities, matching supervised accuracy on many benchmarks. This has transformed how we think about classification — moving from fixed class sets to flexible, language-driven recognition.

## Mathematical Explanation

**Zero-shot learning setup**:
- Seen classes: $S = \{s_1, ..., s_{N_S}\}$ with labeled images
- Unseen classes: $U = \{u_1, ..., u_{N_U}\}$ with only side information
- Test: classify images from $U$ using only semantic descriptions

**Compatibility function**:
$$F(x, c) = \langle \phi(x), \psi(c) \rangle$$

Where $\phi$ is the visual encoder and $\psi$ is the semantic encoder (text).

**Prediction**:
$$\hat{c} = \arg\max_{c \in U} F(x, c)$$

**CLIP-style training**: Maximize cosine similarity between matched image-text pairs:
$$\text{maximize} \frac{\phi(x)^T \psi(t)}{\|\phi(x)\| \|\psi(t)\|}$$

## Code Examples

### Example 1: Zero-shot with CLIP

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Simplified CLIP-like model
class SimpleCLIP(nn.Module):
    def __init__(self, img_dim=512, text_dim=512, embed_dim=256):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, img_dim),
        )
        self.img_proj = nn.Linear(img_dim, embed_dim)
        
        # Text encoder (simplified)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)
    
    def encode_image(self, x):
        features = self.img_encoder(x)
        return F.normalize(self.img_proj(features), dim=-1)
    
    def encode_text(self, text_embeddings):
        return F.normalize(self.text_proj(text_embeddings), dim=-1)
    
    def forward(self, images, texts):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts)
        
        logits = self.logit_scale * img_emb @ txt_emb.T
        return logits

model = SimpleCLIP()

# Example zero-shot classification
class_names = ['cat', 'dog', 'bird', 'fish', 'horse']
# Simulate text embeddings
text_embeds = torch.randn(len(class_names), 512)
text_embeds = F.normalize(text_embeds, dim=-1)

# Encode text (class descriptions)
text_features = model.encode_text(text_embeds)

# Test image
img = torch.randn(1, 3, 224, 224)
img_features = model.encode_image(img)

# Compute similarity
similarity = img_features @ text_features.T
probs = F.softmax(100 * similarity, dim=-1)

print("Zero-shot classification probabilities:")
for name, prob in zip(class_names, probs[0].tolist()):
    print(f"  {name}: {prob:.3f}")

predicted_class = class_names[probs[0].argmax().item()]
print(f"\nPredicted: {predicted_class}")
```

### Example 2: Zero-shot with Attribute Vectors

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Attribute-based zero-shot learning
class AttributeZSL(nn.Module):
    def __init__(self, n_attributes=50, img_dim=512):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, img_dim), nn.ReLU(),
            nn.Linear(img_dim, n_attributes),
        )
    
    def forward(self, x):
        return self.img_encoder(x)

# Define attribute vectors for classes
# Attributes: [has_fur, has_feathers, lives_water, has_legs, can_fly, has_tail, ...]
class_attributes = {
    'cat': torch.tensor([1, 0, 0, 1, 0, 1, 0, 0, 1, 1]),
    'dog': torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 1, 1]),
    'eagle': torch.tensor([0, 1, 0, 1, 1, 1, 0, 0, 1, 0]),
    'goldfish': torch.tensor([0, 0, 1, 0, 0, 1, 0, 1, 0, 0]),
    'horse': torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 0, 1]),
}

model = AttributeZSL(n_attributes=10)

# Predict attributes from image
img = torch.randn(1, 3, 64, 64)
predicted_attrs = torch.sigmoid(model(img))

# Match to nearest class by attribute similarity
best_class = None
best_sim = -1

for name, attrs in class_attributes.items():
    sim = F.cosine_similarity(predicted_attrs, attrs.unsqueeze(0), dim=1)
    if sim.item() > best_sim:
        best_sim = sim.item()
        best_class = name

print(f"Predicted attributes: {(predicted_attrs > 0.5).int().tolist()[0]}")
print(f"Matched to class: {best_class} (sim={best_sim:.3f})")
```

### Example 3: Generalized Zero-shot Learning

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Generalized Zero-Shot Learning (GZSL): classify both seen and unseen
def gzsl_evaluate(model, seen_classes, unseen_classes, images, 
                  text_embeds_seen, text_embeds_unseen):
    """
    Evaluate GZSL: classify among both seen and unseen classes.
    """
    # Combine all class embeddings
    all_text_embeds = torch.cat([text_embeds_seen, text_embeds_unseen])
    all_class_names = seen_classes + unseen_classes
    
    model.eval()
    with torch.no_grad():
        img_embeds = model.encode_image(images)
        similarity = img_embeds @ all_text_embeds.T
        
        # Predictions
        predictions = similarity.argmax(dim=1)
        predicted_classes = [all_class_names[p] for p in predictions]
        
        # Confidence scores
        scores = F.softmax(100 * similarity, dim=-1)
    
    return predicted_classes, scores

# Demo
seen = ['cat', 'dog', 'car']
unseen = ['zebra', 'penguin']

text_seen = F.normalize(torch.randn(len(seen), 256), dim=-1)
text_unseen = F.normalize(torch.randn(len(unseen), 256), dim=-1)

model = SimpleCLIP()
images = torch.randn(4, 3, 224, 224)

preds, scores = gzsl_evaluate(model, seen, unseen, images, 
                              text_seen, text_unseen)

print("GZSL Predictions:")
for i, (p, s) in enumerate(zip(preds, scores)):
    top5_idx = s[0].argsort(descending=True)[:5]
    print(f"  Image {i}: Pred={p}")
    print(f"    Top-3: {[(seen+unseen)[idx] for idx in top5_idx[:3]]}")
```

## Common Mistakes

1. **Bias toward seen classes**: GZSL models tend to predict seen classes over unseen; calibration is needed.
2. **Using incompatible embeddings**: Visual and semantic embeddings must be aligned in the same space.
3. **Poor class descriptions**: Vague or misleading text descriptions hurt performance.
4. **Ignoring the domain gap**: Zero-shot works worse when images differ significantly from training distribution.
5. **Not normalizing embeddings**: Cosine similarity requires normalized embeddings.

## Interview Questions

### Beginner - 5
1. What is zero-shot learning?
2. How does zero-shot differ from few-shot learning?
3. What is a semantic embedding?
4. What is CLIP?
5. What is generalized zero-shot learning?

### Intermediate - 5
1. Explain the compatibility function in ZSL.
2. How does CLIP enable zero-shot classification?
3. What is the bias toward seen classes in GZSL?
4. How do you evaluate zero-shot performance?
5. Compare attribute-based vs text-based ZSL.

### Advanced - 3
1. Design a zero-shot method that handles hierarchical classes.
2. Analyze the domain adaptation aspects of ZSL.
3. Implement a transductive ZSL method that uses unlabeled test data.

## Practice Problems

### Easy - 5
1. Compute cosine similarity between image and text embeddings.
2. Create attribute vectors for 5 animal classes.
3. Implement zero-shot prediction with CLIP.
4. Evaluate zero-shot accuracy on CIFAR-100.
5. Generate class descriptions for zero-shot.

### Medium - 5
1. Train a simple ZSL model on a split of CIFAR.
2. Implement GZSL with calibration.
3. Compare CLIP zero-shot with supervised accuracy.
4. Implement attribute prediction for ZSL.
5. Analyze the effect of text prompt engineering.

### Hard - 3
1. Implement a generative ZSL method (GAZSL).
2. Design a prompt learning method for zero-shot.
3. Analyze the theoretical relationship between zero-shot and few-shot learning.

## Solutions

### Easy - 1 Solution
```python
img_emb = F.normalize(img_emb, dim=-1)
txt_emb = F.normalize(txt_emb, dim=-1)
similarity = img_emb @ txt_emb.T
prediction = class_names[similarity.argmax().item()]
```

## Related Concepts

DL-228 Few-shot Image Classification, DL-230 Open Set Recognition, DL-224 Feature Extraction

## Next Concepts

DL-230 Open Set Recognition

## Summary

Zero-shot image classification enables recognition of unseen classes by leveraging semantic side information (text descriptions or attributes) to bridge the gap between seen and unseen categories. Models like CLIP have made zero-shot classification practical and accurate.

## Key Takeaways

- Zero-shot: classify unseen classes without any labeled examples
- Shared embedding space: images and class descriptions aligned
- CLIP: image-text contrastive learning enables zero-shot
- Attribute-based ZSL: attribute vectors as semantic side information
- Generalized ZSL: classify both seen and unseen classes (harder)
- Bias toward seen classes needs calibration
- Prompt engineering significantly affects performance
- Zero-shot evaluation: unseen class accuracy (ZSL) or harmonic mean (GZSL)
- Applications: open-world recognition, novel object detection
- Represents a paradigm shift from fixed-class to language-driven classification
