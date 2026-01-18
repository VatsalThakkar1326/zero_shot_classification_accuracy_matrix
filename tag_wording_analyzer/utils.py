"""
Utility functions for tag wording analysis.
"""

import pandas as pd
from typing import Dict, List, Optional, Any


def validate_inputs(
    df: pd.DataFrame,
    label_sets: Dict[str, List[str]],
    label_to_canonical: Dict[str, Dict[str, str]],
    hypothesis_templates: Dict[str, str],
) -> None:
    """
    Validate all inputs before running analysis.
    
    Raises:
        ValueError: If any input is invalid.
    """
    # Check DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'text' column")
    
    if "gold" not in df.columns:
        raise ValueError("DataFrame must have a 'gold' column with expected labels")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Check label_sets
    if not isinstance(label_sets, dict) or len(label_sets) == 0:
        raise ValueError("label_sets must be a non-empty dict")
    
    for name, labels in label_sets.items():
        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError(f"label_sets['{name}'] must be a non-empty list")
    
    # Check label_to_canonical
    if not isinstance(label_to_canonical, dict):
        raise ValueError("label_to_canonical must be a dict")
    
    for set_name in label_sets.keys():
        if set_name not in label_to_canonical:
            raise ValueError(f"label_to_canonical missing mapping for '{set_name}'")
        
        # Check all labels have mappings
        labels = dedupe_preserve_order(label_sets[set_name])
        mapping = label_to_canonical[set_name]
        missing = [lab for lab in labels if lab not in mapping]
        if missing:
            raise ValueError(f"label_to_canonical['{set_name}'] missing mappings for: {missing}")
    
    # Check hypothesis_templates
    if not isinstance(hypothesis_templates, dict) or len(hypothesis_templates) == 0:
        raise ValueError("hypothesis_templates must be a non-empty dict")
    
    for name, template in hypothesis_templates.items():
        if "{}" not in template:
            raise ValueError(f"hypothesis_templates['{name}'] must contain '{{}}' placeholder")


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration with example label sets and templates.
    
    Returns:
        Dict with 'label_sets', 'label_to_canonical', 'hypothesis_templates'
    """
    config = {
        "label_sets": {
            "simple": [
                "positive sentiment",
                "negative sentiment",
                "neutral sentiment",
            ],
            "explicit": [
                "The text expresses positive sentiment.",
                "The text expresses negative sentiment.",
                "The text expresses neutral sentiment.",
            ],
        },
        "label_to_canonical": {
            "simple": {
                "positive sentiment": "positive",
                "negative sentiment": "negative",
                "neutral sentiment": "neutral",
            },
            "explicit": {
                "The text expresses positive sentiment.": "positive",
                "The text expresses negative sentiment.": "negative",
                "The text expresses neutral sentiment.": "neutral",
            },
        },
        "hypothesis_templates": {
            "simple": "This text is about {}.",
            "explicit": "The text expresses {}.",
            "classification": "This is {}.",
        },
    }
    return config


def get_canonical_labels(label_to_canonical: Dict[str, Dict[str, str]]) -> List[str]:
    """Extract unique canonical labels from mappings."""
    canonical = set()
    for mapping in label_to_canonical.values():
        canonical.update(mapping.values())
    return sorted(list(canonical))
