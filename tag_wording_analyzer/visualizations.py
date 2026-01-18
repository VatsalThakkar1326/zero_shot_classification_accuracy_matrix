"""
Visualization functions for tag wording analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Any


def plot_results_dashboard(
    results_df: pd.DataFrame,
    all_predictions: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    label_to_canonical: Dict[str, Dict[str, str]],
    show_plot: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 14),
) -> plt.Figure:
    """
    Create the final results dashboard with all visualizations.
    
    Args:
        results_df: DataFrame with analysis results (combo, accuracy, confidence, score)
        all_predictions: Dict of predictions for each combination
        df: Original DataFrame with 'text' and 'gold' columns
        label_to_canonical: Mapping from tag labels to canonical labels
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)
    
    # Get unique label sets and templates for colors/markers
    label_sets = results_df["label_set"].unique().tolist()
    templates = results_df["template"].unique().tolist()
    
    # Create color/marker mappings
    marker_list = ["o", "s", "D", "^", "v", "<", ">", "p", "h"]
    color_list = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    
    markers = {ls: marker_list[i % len(marker_list)] for i, ls in enumerate(label_sets)}
    colors = {t: color_list[i % len(color_list)] for i, t in enumerate(templates)}
    
    # PLOT 1: Bubble Chart - Accuracy vs Confidence
    ax1 = fig.add_subplot(gs[0, :2])
    
    for _, row in results_df.iterrows():
        marker = markers.get(row["label_set"], "o")
        color = colors.get(row["template"], "gray")
        size = row["score"] * 500 + 100
        ax1.scatter(row["accuracy"], row["confidence"], s=size, c=color,
                   marker=marker, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Highlight winner
    winner = results_df.iloc[0]
    ax1.scatter(winner["accuracy"], winner["confidence"], s=winner["score"]*500+100,
               facecolors='none', edgecolors='red', linewidth=3, marker='o')
    ax1.annotate(f"BEST: {winner['combo']}", 
                xy=(winner["accuracy"], winner["confidence"]),
                xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold', color='red')
    
    ax1.set_xlabel("Accuracy", fontsize=11)
    ax1.set_ylabel("Confidence", fontsize=11)
    ax1.set_title("All Combinations: Accuracy vs Confidence\n(Size = Combined Score)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Legend
    legend1 = [Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=10, label=ls) 
               for ls, m in markers.items()]
    legend2 = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=t) 
               for t, c in colors.items()]
    ax1.legend(handles=legend1 + legend2, loc='lower right', fontsize=8, ncol=2)
    
    # PLOT 2: Ranking Bar Chart
    ax2 = fig.add_subplot(gs[0, 2])
    
    top_n = min(8, len(results_df))
    top_combos = results_df.head(top_n)
    bar_colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 else '#4CAF50' 
                  for i in range(top_n)]
    bars = ax2.barh(range(top_n), top_combos['score'], color=bar_colors, edgecolor='black')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([f"{r['label_set']}\n+ {r['template']}" 
                        for _, r in top_combos.iterrows()], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Score (Accuracy x Confidence)", fontsize=10)
    ax2.set_title("Top Combinations", fontsize=11, fontweight='bold')
    for i, (bar, row) in enumerate(zip(bars, top_combos.itertuples())):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{row.score:.2f}", va='center', fontsize=9)
    
    # PLOT 3: Accuracy by Label Set
    ax3 = fig.add_subplot(gs[1, 0])
    
    label_set_summary = results_df.groupby("label_set").agg({"accuracy": "mean", "score": "mean"})
    label_set_summary = label_set_summary.sort_values("score", ascending=True)
    colors_ls = ['#E57373', '#FFB74D', '#81C784', '#64B5F6', '#BA68C8'][:len(label_set_summary)]
    bars3 = ax3.barh(label_set_summary.index, label_set_summary['accuracy'], color=colors_ls, edgecolor='black')
    ax3.set_xlabel("Avg Accuracy", fontsize=10)
    ax3.set_title("By Label Set", fontsize=11, fontweight='bold')
    for bar in bars3:
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{bar.get_width():.1%}", va='center', fontsize=9)
    
    # PLOT 4: Accuracy by Template
    ax4 = fig.add_subplot(gs[1, 1])
    
    template_summary = results_df.groupby("template").agg({"accuracy": "mean", "score": "mean"})
    template_summary = template_summary.sort_values("score", ascending=True)
    colors_tpl = [colors.get(t, 'gray') for t in template_summary.index]
    bars4 = ax4.barh(template_summary.index, template_summary['accuracy'], color=colors_tpl, edgecolor='black')
    ax4.set_xlabel("Avg Accuracy", fontsize=10)
    ax4.set_title("By Template", fontsize=11, fontweight='bold')
    for bar in bars4:
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{bar.get_width():.1%}", va='center', fontsize=9)
    
    # PLOT 5: Heatmap - Label Set x Template
    ax5 = fig.add_subplot(gs[1, 2])
    
    pivot = results_df.pivot(index="label_set", columns="template", values="score")
    im = ax5.imshow(pivot.values, cmap="RdYlGn", aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(range(len(pivot.columns)))
    ax5.set_xticklabels(pivot.columns, fontsize=9, rotation=45, ha='right')
    ax5.set_yticks(range(len(pivot.index)))
    ax5.set_yticklabels(pivot.index, fontsize=9)
    ax5.set_title("Score Heatmap", fontsize=11, fontweight='bold')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax5.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax5, shrink=0.8)
    
    # PLOT 6: Sentence-by-Sentence Results Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Build detailed sentence analysis
    best_combo = results_df.iloc[0]['combo']
    best_preds = all_predictions[best_combo]
    best_ls = results_df.iloc[0]['label_set']
    
    sentence_details = []
    for idx, row in df.iterrows():
        gold = row['gold']
        pred_raw = best_preds.loc[idx, 'pred_label']
        pred_canonical = label_to_canonical[best_ls].get(pred_raw, pred_raw)
        conf = best_preds.loc[idx, 'score_top1']
        correct = "Yes" if pred_canonical == gold else "No"
        
        txt = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
        
        sentence_details.append({
            "Sentence": txt,
            "Expected": gold,
            "Predicted": pred_canonical,
            "Tag Used": pred_raw[:30] + "..." if len(str(pred_raw)) > 30 else pred_raw,
            "Conf": f"{conf:.1%}",
            "Correct": correct
        })
    
    detail_df = pd.DataFrame(sentence_details)
    
    # Create table (limit rows for readability)
    display_df = detail_df.head(15)
    table = ax6.table(cellText=display_df.values, colLabels=display_df.columns,
                      loc='center', cellLoc='center', colColours=['#4CAF50']*len(display_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.6)
    
    # Color results
    for i in range(len(display_df)):
        result = sentence_details[i]['Correct']
        color = '#C8E6C9' if result == "Yes" else '#FFCDD2'
        for j in range(len(display_df.columns)):
            table[(i+1, j)].set_facecolor(color)
    
    if len(df) > 15:
        ax6.set_title(f"Sentence-by-Sentence Results (Showing 15 of {len(df)}, Using: {best_combo})", 
                     fontsize=12, fontweight='bold', pad=20)
    else:
        ax6.set_title(f"Sentence-by-Sentence Results (Using: {best_combo})", 
                     fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle("TAG WORDING SENSITIVITY ANALYSIS - RESULTS", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def print_summary(results_df: pd.DataFrame) -> None:
    """Print a text summary of the analysis results."""
    print("\n" + "=" * 80)
    print("FINAL RANKINGS")
    print("=" * 80)
    
    print("\nTop 10 Combinations:")
    print("-" * 60)
    for i, row in results_df.head(10).iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"  {rank}. {row['combo']}")
        print(f"     Accuracy: {row['accuracy']:.1%} | Confidence: {row['confidence']:.1%} | Score: {row['score']:.3f}")
    
    winner = results_df.iloc[0]
    worst = results_df.iloc[-1]
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"\nBEST COMBINATION: {winner['combo']}")
    print(f"  - Label Set: {winner['label_set']}")
    print(f"  - Template: {winner['template']}")
    print(f"  - Accuracy: {winner['accuracy']:.1%}")
    print(f"  - Confidence: {winner['confidence']:.1%}")
    print(f"  - Combined Score: {winner['score']:.3f}")
    
    print(f"\nWORST COMBINATION: {worst['combo']}")
    print(f"  - Accuracy: {worst['accuracy']:.1%} | Confidence: {worst['confidence']:.1%} | Score: {worst['score']:.3f}")
    
    print("\n" + "-" * 80)
    print("RECOMMENDATION:")
    if winner['accuracy'] >= 0.8:
        print(f"  Use '{winner['label_set']}' tags with '{winner['template']}' template for production.")
    elif winner['accuracy'] >= 0.6:
        print(f"  '{winner['combo']}' works reasonably. Consider adding more explicit tag wordings.")
    else:
        print(f"  Results are weak. Try more descriptive tags or fine-tune the model.")
    print("=" * 80)
