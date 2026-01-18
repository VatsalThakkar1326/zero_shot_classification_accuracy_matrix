"""
Tag Wording Analyzer - Find optimal tag wordings for zero-shot classification models.

Usage:
    from tag_wording_analyzer import analyze_tag_wordings

    results = analyze_tag_wordings(
        df=your_dataframe,  # Must have 'text' and 'gold' columns
        label_sets=your_label_sets,
        label_to_canonical=your_mappings,
        hypothesis_templates=your_templates,
        model_name="MoritzLaurer/bge-m3-zeroshot-v2.0"
    )
"""

from .analyzer import analyze_tag_wordings, TagWordingAnalyzer
from .utils import validate_inputs, create_default_config

__version__ = "1.0.0"
__all__ = ["analyze_tag_wordings", "TagWordingAnalyzer", "validate_inputs", "create_default_config"]
