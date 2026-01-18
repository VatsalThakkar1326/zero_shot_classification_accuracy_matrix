"""
Full working example for tag_wording_analyzer package.
"""

import pandas as pd
from tag_wording_analyzer import analyze_tag_wordings

# =============================================================================
# 1. PREPARE YOUR DATA
# =============================================================================
# DataFrame must have 'text' column and 'gold' column (expected labels)

df = pd.DataFrame({
    "text": [
        "I am so exhausted that I wanna hit the bed.",
        "He injured his knee during the game.",
        "No injuries were reported after the incident.",
        "I want to hurt myself.",
        "I'm feeling much better and not tired today.",
        "My back is killing me after that fall.",
        "She's been feeling suicidal lately.",
        "I slept well and feel energetic this morning.",
        "The accident left him with a broken arm.",
        "I'm so worn out from working all day.",
    ],
    "gold": [
        "tired_positive",
        "injury_positive",
        "injury_negative",
        "self_harm_positive",
        "tired_negative",
        "injury_positive",
        "self_harm_positive",
        "tired_negative",
        "injury_positive",
        "tired_positive",
    ]
})

print(f"Dataset: {len(df)} sentences")
print(df)
print()

# =============================================================================
# 2. DEFINE TAG SETS (different wordings to test)
# =============================================================================
# Each tag set is a list of candidate labels the model will choose from.
# You want to test which WORDING works best.

label_sets = {
    # Messy/simple tags (often don't work well)
    "v1_simple": [
        "Tired positive",
        "Tired negative",
        "Injury positive",
        "Injury negative",
        "Self harm positive",
        "Self harm negative",
    ],
    
    # Clean, natural language tags (usually better)
    "v2_clean": [
        "expresses tiredness or exhaustion",
        "not tired or has energy",
        "mentions an injury or physical harm",
        "no injury mentioned",
        "expresses self-harm or suicidal intent",
        "no self-harm intent",
    ],
    
    # Very explicit, full sentence tags (often best for NLI models)
    "v3_explicit": [
        "The text expresses tiredness or exhaustion.",
        "The text expresses being not tired or rested.",
        "The text reports an injury or physical harm.",
        "The text reports no injury.",
        "The text expresses self-harm or suicidal intent.",
        "The text expresses no self-harm intent.",
    ],
}

# =============================================================================
# 3. MAP TAGS TO CANONICAL LABELS
# =============================================================================
# Each tag must map to one of your canonical labels (the 'gold' values in your df)

label_to_canonical = {
    "v1_simple": {
        "Tired positive": "tired_positive",
        "Tired negative": "tired_negative",
        "Injury positive": "injury_positive",
        "Injury negative": "injury_negative",
        "Self harm positive": "self_harm_positive",
        "Self harm negative": "self_harm_negative",
    },
    "v2_clean": {
        "expresses tiredness or exhaustion": "tired_positive",
        "not tired or has energy": "tired_negative",
        "mentions an injury or physical harm": "injury_positive",
        "no injury mentioned": "injury_negative",
        "expresses self-harm or suicidal intent": "self_harm_positive",
        "no self-harm intent": "self_harm_negative",
    },
    "v3_explicit": {
        "The text expresses tiredness or exhaustion.": "tired_positive",
        "The text expresses being not tired or rested.": "tired_negative",
        "The text reports an injury or physical harm.": "injury_positive",
        "The text reports no injury.": "injury_negative",
        "The text expresses self-harm or suicidal intent.": "self_harm_positive",
        "The text expresses no self-harm intent.": "self_harm_negative",
    },
}

# =============================================================================
# 4. DEFINE HYPOTHESIS TEMPLATES
# =============================================================================
# The {} gets replaced with each candidate label.
# Different templates can significantly affect results!

hypothesis_templates = {
    "simple": "This text is about {}.",
    "explicit": "The text expresses {}.",
    "reporting": "The text reports {}.",
    "is": "This is {}.",
}

# =============================================================================
# 5. RUN THE ANALYSIS
# =============================================================================
# This will test all combinations: 3 tag sets × 4 templates = 12 combinations

print("=" * 60)
print("Running Tag Wording Analysis...")
print("=" * 60)

results = analyze_tag_wordings(
    df=df,
    label_sets=label_sets,
    label_to_canonical=label_to_canonical,
    hypothesis_templates=hypothesis_templates,
    model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",  # Good zero-shot model
    text_col="text",          # Column name for text
    gold_col="gold",          # Column name for expected labels
    batch_size=8,             # Batch size for inference
    show_progress=True,       # Show progress bar
    show_plot=True,           # Display the dashboard
    save_plot="results.png",  # Save figure to file (optional)
    print_results=True,       # Print text summary
)

# =============================================================================
# 6. ACCESS RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\nBest combination: {results.best_combo}")
print(f"Best accuracy: {results.best_accuracy:.1%}")
print(f"Best confidence: {results.best_confidence:.1%}")
print(f"Best score: {results.best_score:.3f}")

print("\n" + "-" * 60)
print("Full rankings:")
print(results.results_df[['combo', 'accuracy', 'confidence', 'score']].to_string())

print("\n" + "-" * 60)
print("Predictions from best combination:")
best_preds = results.all_predictions[results.best_combo]
print(best_preds[['text', 'pred_label', 'pred_canonical', 'score_top1']].to_string())