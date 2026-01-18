# Tag Wording Analysis - Explained Simply

## What Problem Are We Solving?

You have a **zero-shot classification model** (like DeBERTa or BGE-M3) that classifies text into categories. The catch? **The exact wording of your category labels affects the results!**

For example, for the sentence:
> "I am so exhausted that I wanna hit the bed."

The model might classify it differently if you use:
- `"Tired positive"` vs `"expresses tiredness or exhaustion"`
- `"self harm negative"` vs `"no self-harm is mentioned"`

**Goal:** Find which tag wordings work best for your use case.

---

## The Setup

### 1. Your Data
```
Sentences you want to classify + What you EXPECT them to be classified as (gold labels)
```

Example:
| Sentence | Expected Label (Gold) |
|----------|----------------------|
| "I am so exhausted..." | tired_positive |
| "He injured his knee..." | injury_positive |

### 2. Tag Sets (Different Wordings)
You create multiple ways to word the same concept:

| Canonical Label | v1 (messy) | v2 (clean) | v3 (explicit) |
|----------------|------------|------------|---------------|
| tired_positive | "Tired positive" | "expresses tiredness" | "The text expresses tiredness or exhaustion." |
| injury_positive | "Injury Yes" | "mentions an injury" | "The text reports an injury." |

### 3. Hypothesis Templates
The model uses templates like:
- `"This text is about {}."`  → "This text is about tired positive."
- `"The text expresses {}."` → "The text expresses tiredness."

---

## Important Considerations

### 1. Label Distinctiveness (Semantic Similarity)

**Problem:** If your tags are too similar, the model gets confused!

```
BAD (too similar):
- "happy"
- "joyful" 
- "cheerful"
→ Model can't distinguish these!

GOOD (distinct):
- "expresses happiness"
- "expresses sadness"
- "expresses anger"
→ Clear semantic boundaries
```

**How to check:**
- Tags for DIFFERENT classes should be semantically distant
- Tags for the SAME class can be similar (that's fine)
- Avoid synonyms across different canonical classes

### 2. Cross-Validation (Avoid Overfitting)

**Problem:** If you optimize on the same data you test on, you might just memorize what works for THOSE specific sentences, not sentences in general.

**Solution: Split your data:**
```
Your sentences (100 total)
    ├── Development Set (70%) → Use to find best combo
    └── Validation Set (30%) → Use to confirm it generalizes
```

**The process:**
1. Find best combo using Development Set
2. Test that combo on Validation Set (unseen data)
3. If performance drops significantly → You overfit!
4. If performance stays similar → Your result is reliable

### 3. Overconfident Errors

**Problem:** A model that's 95% confident but WRONG is more dangerous than one that's 50% confident and wrong.

**Solution: Track confidence on errors:**
```
Avg Confidence (when WRONG) = Sum of wrong prediction confidences / Number of wrong predictions
```

- Low confidence on errors = Good (model knows when it's unsure)
- High confidence on errors = Bad (model is overconfident)

---

## The Math (Simplified)

### Step 1: Run Predictions
For each combination of (tag set × template), run the model on all sentences.

```
12 combinations = 3 tag sets × 4 templates
```

The model gives you:
- **Predicted label** (what it thinks the sentence is about)
- **Confidence score** (how sure it is, 0 to 1)

### Step 2: Calculate Accuracy

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

Example: If model gets 4 out of 5 sentences right:
```
Accuracy = 4/5 = 0.80 = 80%
```

### Step 3: Calculate Average Confidence (When Correct)

We only care about confidence when the model got it RIGHT:

```
Avg Confidence = Sum of confidence scores for correct predictions / Number of correct predictions
```

Example: Model correctly predicted 3 sentences with confidences 0.9, 0.8, 0.7:
```
Avg Confidence = (0.9 + 0.8 + 0.7) / 3 = 0.80
```

### Step 4: Combined Score

This is the key metric that balances accuracy AND confidence:

```
Combined Score = Accuracy × Average Confidence
```

**Why multiply?**
- High accuracy + Low confidence = Not reliable (lucky guesses)
- Low accuracy + High confidence = Confidently wrong (bad!)
- High accuracy + High confidence = WINNER

Example:
| Combo | Accuracy | Confidence | Combined Score |
|-------|----------|------------|----------------|
| A | 100% | 0.50 | 0.50 |
| B | 80% | 0.90 | 0.72 |
| C | 60% | 0.95 | 0.57 |

**Combo B wins** - good accuracy with high confidence!

#### Alternative Scoring Methods

The simple multiplication treats accuracy and confidence equally. You might prefer:

**1. Weighted Combination (tunable emphasis):**
```
Weighted Score = (α × Accuracy) + ((1-α) × Confidence)
```
- α = 0.7 → Emphasize accuracy more
- α = 0.3 → Emphasize confidence more
- α = 0.5 → Equal weight (default)

**2. Harmonic Mean (penalizes imbalance):**
```
Harmonic Score = 2 × (Accuracy × Confidence) / (Accuracy + Confidence)
```
This punishes cases where one metric is high but the other is low.

**3. Track Overconfident Errors:**
```
Overconfidence Penalty = Avg confidence on WRONG predictions
```
High confidence on wrong predictions is dangerous! Subtract this from your score.

---

## What Each Analysis Shows

### 1. Ranking Table
Lists all combinations sorted by their performance (Macro F1 or Combined Score).

### 2. Confusion Matrix
A grid showing:
- **Rows:** What you expected (gold label)
- **Columns:** What the model predicted

```
                    Predicted →
Expected ↓    injury_pos  tired_pos  self_harm_pos
injury_pos         5          0            1
tired_pos          0          4            0
self_harm_pos      0          1            3
```
- Diagonal (5, 4, 3) = Correct predictions
- Off-diagonal (1, 1) = Mistakes

### 3. Transition Matrix (Flip Analysis)
Shows how predictions CHANGE when you switch tag wordings:

```
When using v1 tags:  "I am exhausted..." → tired_NEGATIVE (wrong!)
When using v2 tags:  "I am exhausted..." → tired_POSITIVE (correct!)
```

This reveals which wordings the model is sensitive to.

### 4. Margin Analysis
**Margin = Top1 confidence - Top2 confidence**

```
If model says:
  - tired_positive: 0.45
  - self_harm_positive: 0.42
  
Margin = 0.45 - 0.42 = 0.03 (very small!)
```

Small margin = Model is UNSURE → These are the cases where wording changes matter most!

### 5. Heatmap
Visual grid showing performance of each (tag set × template) combination:
- **Dark green** = High score (good)
- **Light/white** = Low score (bad)

---

## Key Metrics Explained

### Macro F1 Score
Combines precision and recall, averaged across all classes:

```
Precision = Correct predictions for class X / All predictions of class X
Recall = Correct predictions for class X / All actual class X samples
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Macro F1 = Average F1 across all classes
```

**Why Macro F1?** It treats all classes equally, even if some have fewer samples.

### Confidence Score
The model's self-reported certainty (0 to 1):
- **0.95** = "I'm 95% sure this is tired_positive"
- **0.30** = "I'm only 30% sure..."

---

## How to Interpret Results

### Good Signs
1. Combined score > 0.7
2. Accuracy > 80%
3. Confidence > 0.7 when correct
4. Few "flip cases" between tag sets
5. Large margins (model is decisive)

### Bad Signs
1. Combined score < 0.3
2. Many flip cases (model is sensitive to wording)
3. Small margins (model is unsure)
4. Off-diagonal errors in confusion matrix
5. **High confidence on WRONG predictions** (overconfident errors)
6. **Semantically similar tags across different classes** (label confusion)

---

## Best Practices for Reliable Results

### 1. Use Train/Validation Split
```python
# Split your data
from sklearn.model_selection import train_test_split
df_dev, df_val = train_test_split(df, test_size=0.3, stratify=df['gold'])

# Find best combo on df_dev
# Confirm on df_val
```

### 2. Check Label Distinctiveness
Before running analysis, verify your tags are semantically distinct:
- "injury" vs "tiredness" → Good (different concepts)
- "injury yes" vs "injury positive" → Bad if in same tag set (redundant)
- "tired" vs "exhausted" → OK if both map to same canonical class

### 3. Monitor Overconfidence
```
Good model:
  - Correct predictions: High confidence (0.8+)
  - Wrong predictions: Low confidence (0.3-)

Bad model:
  - Correct predictions: Medium confidence (0.6)
  - Wrong predictions: Also medium confidence (0.6)
  → Model doesn't know what it doesn't know!
```

---

## The Final Verdict Logic

```python
# For each combination:
1. Count correct predictions
2. Calculate accuracy
3. Calculate average confidence (when correct)
4. Combined score = accuracy × confidence

# Pick the winner:
Best combo = Highest combined score
```

---

## Simple Example Walkthrough

**Sentence:** "I am so exhausted that I wanna hit the bed."
**Expected:** tired_positive

| Combo | Predicted | Correct? | Confidence | Tag Used |
|-------|-----------|----------|------------|----------|
| v1 + simple | tired_negative | No | 0.33 | "Tired Negative" |
| v2 + simple | tired_positive | Yes | 0.89 | "expresses tiredness" |
| v3 + simple | tired_positive | Yes | 0.94 | "The text expresses tiredness" |

**Conclusion:** 
- v1 tags FAIL for this sentence (confusing wording)
- v3 tags work BEST (explicit, natural language)

---

## TL;DR (Too Long; Didn't Read)

1. **Try different tag wordings** for the same concepts
2. **Run model on your test sentences** with each wording
3. **Measure accuracy + confidence** for each combination
4. **Pick the wording** with highest combined score
5. **Use that wording** in production!

The whole notebook automates this process and shows you exactly which wordings work best for YOUR data.
