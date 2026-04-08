#!/usr/bin/env python3
"""Generate the three-condition comparison figure for the DSH paper.

Creates fig_three_conditions.pdf showing:
- Panel A: Mean effect sizes by model and prompt condition
- Panel B: Percentage of markers significantly altered
- Panel C: Key voice markers across conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
RESULTS_DIR = Path("results")
PAPER_DIR = Path("Narrative_Normalization/paper/dsh")

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color palette for three conditions
COLORS = {
    'generic': '#2c3e50',      # Dark blue-gray
    'rewrite_only': '#7f8c8d', # Medium gray
    'voice_preserving': '#bdc3c7',  # Light gray
}

MODEL_LABELS = {
    'gpt54': 'GPT-5.4',
    'claude_sonnet': 'Claude Sonnet 4.6',
    'gemini_31_pro': 'Gemini 3.1 Pro',
}

CONDITION_LABELS = {
    'generic': 'Generic\n("improve")',
    'rewrite_only': 'Rewrite-only\n("rewrite")',
    'voice_preserving': 'Voice-preserving',
}


def load_all_comparisons():
    """Load comparison data for all conditions and models."""
    conditions = ['generic', 'voice_preserving', 'rewrite_only']
    models = ['gpt54', 'claude_sonnet', 'gemini_31_pro']

    data = {}
    for cond in conditions:
        data[cond] = {}
        for model in models:
            try:
                df = pd.read_csv(RESULTS_DIR / f"comparison_{cond}_{model}.csv")
                data[cond][model] = df
            except FileNotFoundError:
                print(f"Warning: Missing {cond}_{model}")

    return data


def compute_summary_stats(data):
    """Compute summary statistics for each condition/model combo."""
    stats = []
    for cond, models_data in data.items():
        for model, df in models_data.items():
            sig_count = df['significant_fdr'].sum()
            total = len(df)
            pct_sig = 100 * sig_count / total
            mean_d = df['cohens_d'].abs().mean()

            stats.append({
                'condition': cond,
                'model': model,
                'pct_significant': pct_sig,
                'mean_abs_d': mean_d,
                'n_significant': sig_count,
                'n_total': total,
            })

    return pd.DataFrame(stats)


def plot_three_conditions(data, stats_df):
    """Create the three-panel comparison figure."""
    fig = plt.figure(figsize=(14, 10), dpi=150)

    # Create grid for three panels
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])  # Panel A: Effect sizes
    ax2 = fig.add_subplot(gs[0, 1])  # Panel B: % Significant
    ax3 = fig.add_subplot(gs[1, :])  # Panel C: Key markers (full width)

    models = ['gpt54', 'claude_sonnet', 'gemini_31_pro']
    conditions = ['generic', 'rewrite_only', 'voice_preserving']

    # ── Panel A: Mean Effect Sizes ──
    x = np.arange(len(models))
    width = 0.25

    for i, cond in enumerate(conditions):
        values = [stats_df[(stats_df['condition'] == cond) &
                          (stats_df['model'] == m)]['mean_abs_d'].values[0]
                  for m in models]
        bars = ax1.bar(x + i*width - width, values, width,
                       label=CONDITION_LABELS[cond],
                       color=COLORS[cond], edgecolor='#222222', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax1.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel('Mean |Cohen\'s d|', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_title('(A) Mean Effect Size by Condition', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # ── Panel B: Percentage Significant ──
    for i, cond in enumerate(conditions):
        values = [stats_df[(stats_df['condition'] == cond) &
                          (stats_df['model'] == m)]['pct_significant'].values[0]
                  for m in models]
        bars = ax2.bar(x + i*width - width, values, width,
                       label=CONDITION_LABELS[cond],
                       color=COLORS[cond], edgecolor='#222222', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax2.annotate(f'{val:.0f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('% Markers Significantly Altered', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax2.set_title('(B) Proportion of Markers Altered', fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # ── Panel C: Key Voice Markers ──
    key_markers = [
        'voice_contraction_density',
        'ld_mtld',
        'voice_first_person_density',
        'voice_disfluency',
        'read_flesch_ease',
    ]
    marker_labels = [
        'Contraction\nDensity',
        'Lexical Diversity\n(MTLD)',
        'First-Person\nDensity',
        'Disfluency\nMarkers',
        'Readability\n(Flesch Ease)',
    ]

    # Get effect sizes for GPT-5.4 (representative model) across conditions
    x_markers = np.arange(len(key_markers))

    for i, cond in enumerate(conditions):
        df = data[cond]['gpt54']
        values = []
        for marker in key_markers:
            row = df[df['marker'] == marker]
            if len(row) > 0:
                values.append(row['cohens_d'].values[0])
            else:
                values.append(0)

        ax3.bar(x_markers + i*width - width, values, width,
               label=CONDITION_LABELS[cond],
               color=COLORS[cond], edgecolor='#222222', linewidth=0.5)

    ax3.axhline(0, color='#222222', linewidth=1)
    ax3.set_ylabel("Cohen's d (GPT-5.4)", fontweight='bold')
    ax3.set_xticks(x_markers)
    ax3.set_xticklabels(marker_labels)
    ax3.set_title('(C) Key Voice Markers: Effect Sizes Across Conditions',
                  fontweight='bold', loc='left')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation about generic ≈ rewrite
    ax3.annotate('Generic and Rewrite-only show\nnearly identical patterns',
                xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=9, fontstyle='italic', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='none'))

    fig.suptitle('Normalization Across Three Prompt Conditions',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig


def main():
    print("Loading comparison data...")
    data = load_all_comparisons()

    print("Computing summary statistics...")
    stats_df = compute_summary_stats(data)

    print("\nSummary by condition:")
    summary = stats_df.groupby('condition').agg({
        'pct_significant': 'mean',
        'mean_abs_d': 'mean'
    }).round(2)
    print(summary)

    print("\nGenerating figure...")
    fig = plot_three_conditions(data, stats_df)

    # Save to paper directory
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_DIR / "fig_three_conditions.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(PAPER_DIR / "fig_three_conditions.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {PAPER_DIR / 'fig_three_conditions.pdf'}")

    # Also save to results for reference
    fig.savefig(RESULTS_DIR / "fig_three_conditions.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(RESULTS_DIR / "fig_three_conditions.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {RESULTS_DIR / 'fig_three_conditions.pdf'}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
