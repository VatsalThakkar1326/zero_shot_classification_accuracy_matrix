# Tag Wording Analyzer

A Python library and web app for finding optimal tag wordings for zero-shot classification models.

## Problem

Zero-shot classification models (like DeBERTa, BGE-M3) are sensitive to **how you word your labels**. The same concept can get wildly different results:

| Tag Wording | Accuracy |
|-------------|----------|
| "Tired positive" | 10% |
| "expresses tiredness or exhaustion" | 100% |

This tool helps you **find which wording works best** for your use case.

---

## Installation

### 1. Clone/Download the repository

```bash
cd quantitative_analysis
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. CUDA Setup (GPU Acceleration)

For faster inference, install PyTorch with CUDA support:

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA is working:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**No GPU?** The library works on CPU too, just slower. Set `device=-1` when calling the analyzer.

---

## Quick Start (Python Library)

```python
import pandas as pd
from tag_wording_analyzer import analyze_tag_wordings

# 1. Your data (text + expected labels)
df = pd.DataFrame({
    "text": [
        "I am so exhausted that I wanna hit the bed.",
        "He injured his knee during the game.",
        "I want to hurt myself.",
    ],
    "gold": [
        "tired_positive",
        "injury_positive",
        "self_harm_positive",
    ]
})

# 2. Different tag wordings to test
label_sets = {
    "simple": ["tired", "injury", "self-harm"],
    "explicit": ["expresses tiredness", "mentions injury", "self-harm intent"],
}

# 3. Map tags to your canonical labels
label_to_canonical = {
    "simple": {
        "tired": "tired_positive",
        "injury": "injury_positive",
        "self-harm": "self_harm_positive",
    },
    "explicit": {
        "expresses tiredness": "tired_positive",
        "mentions injury": "injury_positive",
        "self-harm intent": "self_harm_positive",
    },
}

# 4. Hypothesis templates
hypothesis_templates = {
    "simple": "This text is about {}.",
    "explicit": "The text expresses {}.",
}

# 5. Run analysis
results = analyze_tag_wordings(
    df=df,
    label_sets=label_sets,
    label_to_canonical=label_to_canonical,
    hypothesis_templates=hypothesis_templates,
    model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",
)

# 6. Get results
print(f"Best: {results.best_combo}")
print(f"Accuracy: {results.best_accuracy:.1%}")
print(results.results_df)
```

---

## Streamlit Web App

For a no-code interface:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Features:
- Upload CSV or enter data manually
- Define label sets visually
- Edit mappings with JSON editor
- Run analysis with one click
- Download results as CSV
- Interactive visualizations

---

## How It Works

1. **You provide:**
   - Sentences to classify
   - Expected labels (gold)
   - Different tag wordings to test
   - Hypothesis templates

2. **The analyzer:**
   - Runs all combinations (tag set × template)
   - Measures accuracy against your gold labels
   - Measures model confidence
   - Calculates combined score = accuracy × confidence

3. **You get:**
   - Ranking of all combinations
   - Best tag wording for your use case
   - Visualizations (heatmaps, charts)
   - Per-sentence predictions

---

## API Reference

### `analyze_tag_wordings()`

Main function to run the analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Must have 'text' and 'gold' columns |
| `label_sets` | dict | required | Tag wordings to test |
| `label_to_canonical` | dict | required | Maps tags → canonical labels |
| `hypothesis_templates` | dict | required | Templates with `{}` placeholder |
| `model_name` | str | "MoritzLaurer/bge-m3-zeroshot-v2.0" | HuggingFace model |
| `text_col` | str | "text" | Column name for text |
| `gold_col` | str | "gold" | Column name for expected labels |
| `batch_size` | int | 8 | Inference batch size |
| `show_plot` | bool | True | Display dashboard |
| `save_plot` | str | None | Path to save figure |
| `print_results` | bool | True | Print text summary |
| `device` | int | None | 0=GPU, -1=CPU, None=auto |

### Returns: `AnalysisResults`

| Attribute | Type | Description |
|-----------|------|-------------|
| `results_df` | DataFrame | All combinations ranked |
| `all_predictions` | dict | Predictions for each combo |
| `best_combo` | str | Name of best combination |
| `best_accuracy` | float | Winner's accuracy |
| `best_confidence` | float | Winner's confidence |
| `best_score` | float | Winner's combined score |
| `figure` | Figure | Matplotlib figure object |

---

## Project Structure

```
quantitative_analysis/
├── tag_wording_analyzer/       # Python package
│   ├── __init__.py
│   ├── analyzer.py             # Core logic
│   ├── visualizations.py       # Plotting
│   ├── utils.py                # Helpers
│   └── README.md
├── app.py                      # Streamlit web app
├── test.py                     # Example script
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── ANALYSIS_EXPLAINED.md       # Detailed explanation
└── README.md                   # This file
```

---

## Recommended Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `MoritzLaurer/bge-m3-zeroshot-v2.0` | Fast | Excellent | Recommended default |
| `facebook/bart-large-mnli` | Medium | Good | Classic choice |
| `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | Slow | Excellent | Best quality |

---

## Tips for Best Results

1. **Use natural language tags** - "expresses tiredness" beats "tired_positive"
2. **Keep tags semantically distinct** - Don't use synonyms for different classes
3. **Test multiple templates** - Some models prefer "This is {}" vs "The text expresses {}"
4. **Add more test sentences** - 20-50 per class is ideal
5. **Look at low-margin cases** - These reveal model sensitivity

---

## License

MIT License
