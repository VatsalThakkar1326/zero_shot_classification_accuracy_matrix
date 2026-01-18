"""
Core analyzer module for tag wording sensitivity analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm.auto import tqdm

from .utils import validate_inputs, dedupe_preserve_order, get_canonical_labels
from .visualizations import plot_results_dashboard, print_summary


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    results_df: pd.DataFrame
    all_predictions: Dict[str, pd.DataFrame]
    best_combo: str
    best_accuracy: float
    best_confidence: float
    best_score: float
    figure: Optional[Any] = None
    
    def __repr__(self):
        return (f"AnalysisResults(\n"
                f"  best_combo='{self.best_combo}',\n"
                f"  best_accuracy={self.best_accuracy:.1%},\n"
                f"  best_confidence={self.best_confidence:.1%},\n"
                f"  best_score={self.best_score:.3f},\n"
                f"  num_combinations={len(self.results_df)}\n"
                f")")


class TagWordingAnalyzer:
    """
    Analyzer for finding optimal tag wordings for zero-shot classification models.
    
    Example:
        analyzer = TagWordingAnalyzer(
            label_sets=my_label_sets,
            label_to_canonical=my_mappings,
            hypothesis_templates=my_templates,
            model_name="MoritzLaurer/bge-m3-zeroshot-v2.0"
        )
        results = analyzer.analyze(df)
    """
    
    def __init__(
        self,
        label_sets: Dict[str, List[str]],
        label_to_canonical: Dict[str, Dict[str, str]],
        hypothesis_templates: Dict[str, str],
        model_name: str = "MoritzLaurer/bge-m3-zeroshot-v2.0",
        device: Optional[int] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the analyzer.
        
        Args:
            label_sets: Dict mapping set names to lists of tag labels
            label_to_canonical: Dict mapping set names to {tag: canonical_label} dicts
            hypothesis_templates: Dict mapping template names to template strings with {}
            model_name: Hugging Face model name for zero-shot classification
            device: Device to use (0 for GPU, -1 for CPU, None for auto)
            use_fp16: Whether to use FP16 on GPU for faster inference
        """
        self.label_sets = label_sets
        self.label_to_canonical = label_to_canonical
        self.hypothesis_templates = hypothesis_templates
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        # Auto-detect device
        if device is None:
            import torch
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        
        self.classifier = None
        self.canonical_labels = get_canonical_labels(label_to_canonical)
    
    def _load_model(self) -> None:
        """Load the zero-shot classification model."""
        if self.classifier is not None:
            return
        
        from transformers import pipeline
        import torch
        
        model_kwargs = {}
        if self.device == 0 and self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
    
    def _run_predictions(
        self,
        texts: List[str],
        label_set_name: str,
        hypothesis_template: str,
        batch_size: int = 8,
    ) -> pd.DataFrame:
        """Run zero-shot predictions for a single combination."""
        self._load_model()
        
        raw_labels = self.label_sets[label_set_name]
        labels = dedupe_preserve_order(raw_labels)
        mapping = self.label_to_canonical[label_set_name]
        
        rows = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            outs = self.classifier(
                batch,
                candidate_labels=labels,
                hypothesis_template=hypothesis_template,
                multi_label=False,
            )
            
            if isinstance(outs, dict):
                outs = [outs]
            
            for txt, out in zip(batch, outs):
                lab_sorted = out["labels"]
                score_sorted = out["scores"]
                top1_label = lab_sorted[0]
                top1 = float(score_sorted[0])
                top2 = float(score_sorted[1]) if len(score_sorted) > 1 else float("nan")
                margin = top1 - top2 if len(score_sorted) > 1 else float("nan")
                
                topk = [(lab, float(sc)) for lab, sc in list(zip(lab_sorted, score_sorted))[:5]]
                
                rows.append({
                    "text": txt,
                    "pred_label": top1_label,
                    "pred_canonical": mapping.get(top1_label, "other"),
                    "score_top1": top1,
                    "score_top2": top2,
                    "margin": margin,
                    "topk": topk,
                })
        
        return pd.DataFrame(rows)
    
    def analyze(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        gold_col: str = "gold",
        batch_size: int = 8,
        show_progress: bool = True,
        show_plot: bool = True,
        save_plot: Optional[str] = None,
        print_results: bool = True,
    ) -> AnalysisResults:
        """
        Run the full tag wording sensitivity analysis.
        
        Args:
            df: DataFrame with text and gold label columns
            text_col: Name of the text column
            gold_col: Name of the gold/expected label column
            batch_size: Batch size for model inference
            show_progress: Whether to show progress bars
            show_plot: Whether to display the results dashboard
            save_plot: Optional path to save the figure
            print_results: Whether to print the text summary
            
        Returns:
            AnalysisResults object with all results
        """
        # Prepare DataFrame with standard column names
        analysis_df = df.rename(columns={text_col: "text", gold_col: "gold"})
        
        # Validate inputs
        validate_inputs(
            analysis_df,
            self.label_sets,
            self.label_to_canonical,
            self.hypothesis_templates,
        )
        
        texts = analysis_df["text"].tolist()
        golds = analysis_df["gold"].tolist()
        
        # Run all combinations
        all_predictions = {}
        combinations = [
            (ls_name, tpl_name)
            for ls_name in self.label_sets.keys()
            for tpl_name in self.hypothesis_templates.keys()
        ]
        
        iterator = tqdm(combinations, desc="Analyzing combinations") if show_progress else combinations
        
        for ls_name, tpl_name in iterator:
            key = f"{ls_name}__{tpl_name}"
            template = self.hypothesis_templates[tpl_name]
            
            preds = self._run_predictions(texts, ls_name, template, batch_size)
            all_predictions[key] = preds
        
        # Calculate results for each combination
        results_rows = []
        for key, preds in all_predictions.items():
            parts = key.split("__")
            ls_name, tpl_name = parts[0], parts[1]
            
            pred_canonical = preds["pred_label"].map(self.label_to_canonical[ls_name])
            correct_mask = (pred_canonical == analysis_df["gold"])
            accuracy = correct_mask.mean()
            avg_conf = preds["score_top1"].mean()
            
            conf_correct = preds.loc[correct_mask, "score_top1"].mean() if correct_mask.sum() > 0 else 0
            conf_wrong = preds.loc[~correct_mask, "score_top1"].mean() if (~correct_mask).sum() > 0 else 0
            
            results_rows.append({
                "combo": key,
                "label_set": ls_name,
                "template": tpl_name,
                "accuracy": accuracy,
                "confidence": avg_conf,
                "conf_correct": conf_correct,
                "conf_wrong": conf_wrong,
                "score": accuracy * avg_conf,
            })
        
        results_df = pd.DataFrame(results_rows).sort_values("score", ascending=False).reset_index(drop=True)
        
        # Create visualizations
        fig = None
        if show_plot or save_plot:
            fig = plot_results_dashboard(
                results_df=results_df,
                all_predictions=all_predictions,
                df=analysis_df,
                label_to_canonical=self.label_to_canonical,
                show_plot=show_plot,
                save_path=save_plot,
            )
        
        # Print summary
        if print_results:
            print_summary(results_df)
        
        # Build results object
        winner = results_df.iloc[0]
        results = AnalysisResults(
            results_df=results_df,
            all_predictions=all_predictions,
            best_combo=winner["combo"],
            best_accuracy=winner["accuracy"],
            best_confidence=winner["confidence"],
            best_score=winner["score"],
            figure=fig,
        )
        
        return results


def analyze_tag_wordings(
    df: pd.DataFrame,
    label_sets: Dict[str, List[str]],
    label_to_canonical: Dict[str, Dict[str, str]],
    hypothesis_templates: Dict[str, str],
    model_name: str = "MoritzLaurer/bge-m3-zeroshot-v2.0",
    text_col: str = "text",
    gold_col: str = "gold",
    batch_size: int = 8,
    show_progress: bool = True,
    show_plot: bool = True,
    save_plot: Optional[str] = None,
    print_results: bool = True,
    device: Optional[int] = None,
) -> AnalysisResults:
    """
    Analyze tag wording sensitivity for zero-shot classification.
    
    This is the main entry point for the library. Pass your data and configurations,
    and get back a complete analysis with visualizations.
    
    Args:
        df: DataFrame with text and gold label columns
        label_sets: Dict mapping set names to lists of tag labels
            Example: {"clean": ["mentions injury", "no injury"], ...}
        label_to_canonical: Dict mapping set names to {tag: canonical_label} dicts
            Example: {"clean": {"mentions injury": "injury_positive", ...}, ...}
        hypothesis_templates: Dict mapping template names to template strings
            Example: {"simple": "This text is about {}.", ...}
        model_name: Hugging Face model name for zero-shot classification
        text_col: Name of the text column in df
        gold_col: Name of the gold/expected label column in df
        batch_size: Batch size for model inference
        show_progress: Whether to show progress bars
        show_plot: Whether to display the results dashboard
        save_plot: Optional path to save the figure
        print_results: Whether to print the text summary
        device: Device to use (0 for GPU, -1 for CPU, None for auto)
        
    Returns:
        AnalysisResults object containing:
            - results_df: DataFrame with all combination scores
            - all_predictions: Dict of predictions for each combination
            - best_combo: Name of the best combination
            - best_accuracy: Accuracy of the best combination
            - best_confidence: Confidence of the best combination
            - best_score: Combined score of the best combination
            - figure: matplotlib Figure object (if show_plot or save_plot)
    
    Example:
        >>> import pandas as pd
        >>> from tag_wording_analyzer import analyze_tag_wordings
        >>> 
        >>> df = pd.DataFrame({
        ...     "text": ["I am so tired...", "He injured his knee..."],
        ...     "gold": ["tired_positive", "injury_positive"]
        ... })
        >>> 
        >>> label_sets = {
        ...     "simple": ["tired", "injury", "other"],
        ...     "explicit": ["expresses tiredness", "mentions injury", "other"]
        ... }
        >>> 
        >>> label_to_canonical = {
        ...     "simple": {"tired": "tired_positive", "injury": "injury_positive", "other": "other"},
        ...     "explicit": {"expresses tiredness": "tired_positive", "mentions injury": "injury_positive", "other": "other"}
        ... }
        >>> 
        >>> templates = {
        ...     "basic": "This text is about {}.",
        ...     "explicit": "The text expresses {}."
        ... }
        >>> 
        >>> results = analyze_tag_wordings(
        ...     df=df,
        ...     label_sets=label_sets,
        ...     label_to_canonical=label_to_canonical,
        ...     hypothesis_templates=templates
        ... )
        >>> 
        >>> print(results.best_combo)
        >>> print(results.best_accuracy)
    """
    analyzer = TagWordingAnalyzer(
        label_sets=label_sets,
        label_to_canonical=label_to_canonical,
        hypothesis_templates=hypothesis_templates,
        model_name=model_name,
        device=device,
    )
    
    return analyzer.analyze(
        df=df,
        text_col=text_col,
        gold_col=gold_col,
        batch_size=batch_size,
        show_progress=show_progress,
        show_plot=show_plot,
        save_plot=save_plot,
        print_results=print_results,
    )
