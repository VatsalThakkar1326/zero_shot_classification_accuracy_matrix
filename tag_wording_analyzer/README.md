# Tag Wording Analyzer

A Python library for finding optimal tag wordings for zero-shot classification models.

## Installation

```bash
# From the project directory
pip install -e .

# Or install dependencies directly
pip install transformers torch pandas numpy matplotlib scikit-learn tqdm
```

## Quick Start

```python
import pandas as pd
from tag_wording_analyzer import analyze_tag_wordings

# 1. Prepare your data
df = pd.DataFrame({
    "text": [
        "I am so exhausted that I wanna hit the bed.",
        "He injured his knee during the game.",
        "No injuries were reported after the incident.",
        "I want to hurt myself.",
        "I'm feeling much better and not tired today."
    ],
    "gold": [
        "tired_positive",
        "injury_positive",
        "injury_negative",
        "self_harm_positive",
        "tired_negative"
    ]
})

# 2. Define your tag sets (different wordings to test)
label_sets = {
    "simple": [
        "tired",
        "not tired",
        "injury",
        "no injury",
        "self-harm",
        "no self-harm",
    ],
    "explicit": [
        "expresses tiredness or exhaustion",
        "not tired or has energy",
        "mentions an injury",
        "no injury mentioned",
        "expresses self-harm intent",
        "no self-harm intent",
    ],
}

# 3. Map each tag to your canonical labels
label_to_canonical = {
    "simple": {
        "tired": "tired_positive",
        "not tired": "tired_negative",
        "injury": "injury_positive",
        "no injury": "injury_negative",
        "self-harm": "self_harm_positive",
        "no self-harm": "self_harm_negative",
    },
    "explicit": {
        "expresses tiredness or exhaustion": "tired_positive",
        "not tired or has energy": "tired_negative",
        "mentions an injury": "injury_positive",
        "no injury mentioned": "injury_negative",
        "expresses self-harm intent": "self_harm_positive",
        "no self-harm intent": "self_harm_negative",
    },
}

# 4. Define hypothesis templates
hypothesis_templates = {
    "simple": "This text is about {}.",
    "explicit": "The text expresses {}.",
    "classification": "This is {}.",
}

# 5. Run the analysis
results = analyze_tag_wordings(
    df=df,
    label_sets=label_sets,
    label_to_canonical=label_to_canonical,
    hypothesis_templates=hypothesis_templates,
    model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",  # or any zero-shot model
)

# 6. Access results
print(f"Best combination: {results.best_combo}")
print(f"Best accuracy: {results.best_accuracy:.1%}")
print(f"Best score: {results.best_score:.3f}")

# Full results DataFrame
print(results.results_df)
```

## Using the Class-Based API

For more control, use the `TagWordingAnalyzer` class:

```python
from tag_wording_analyzer import TagWordingAnalyzer

# Create analyzer instance (model loads lazily)
analyzer = TagWordingAnalyzer(
    label_sets=label_sets,
    label_to_canonical=label_to_canonical,
    hypothesis_templates=hypothesis_templates,
    model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",
    device=0,  # 0 for GPU, -1 for CPU
)

# Run analysis
results = analyzer.analyze(
    df=df,
    show_plot=True,
    save_plot="results.png",
    print_results=True,
)

# Reuse analyzer for different data
results2 = analyzer.analyze(df=another_df)
```

## Output

The analysis produces:

1. **Visual Dashboard** with:
   - Bubble chart (accuracy vs confidence)
   - Ranking bar chart
   - Performance by label set
   - Performance by template
   - Heatmap (label set x template)
   - Sentence-by-sentence results table

2. **Results Object** containing:
   - `results_df`: Full DataFrame with all scores
   - `all_predictions`: Predictions for each combination
   - `best_combo`: Name of winning combination
   - `best_accuracy`: Winner's accuracy
   - `best_confidence`: Winner's confidence
   - `best_score`: Winner's combined score (accuracy x confidence)

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | Must have text and gold label columns |
| `label_sets` | Dict[str, List[str]] | Tag wordings to test |
| `label_to_canonical` | Dict[str, Dict[str, str]] | Maps tags to canonical labels |
| `hypothesis_templates` | Dict[str, str] | Templates with `{}` placeholder |
| `model_name` | str | HuggingFace model name |
| `text_col` | str | Name of text column (default: "text") |
| `gold_col` | str | Name of gold label column (default: "gold") |
| `batch_size` | int | Inference batch size (default: 8) |
| `show_plot` | bool | Display the dashboard (default: True) |
| `save_plot` | str | Path to save figure (optional) |
| `print_results` | bool | Print text summary (default: True) |
| `device` | int | 0=GPU, -1=CPU, None=auto |

## Requirements

- Python 3.8+
- transformers
- torch
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm
