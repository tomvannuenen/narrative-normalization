#!/usr/bin/env python3
"""Generate the main narrative normalization figure for the DSH paper.

Creates fig_narrative_normalization.pdf showing:
- Panel A: Heatmap of effect sizes by dimension
- Panel B: Horizontal bar chart (inflation vs. deflation)
- Panel C: Contraction density by model and condition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Setup paths (relative to repo root, run from repo root directory)
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Publication settings (matching generate_three_condition_figure.py)
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
    'generic': '#333333',      # Dark gray
    'rewrite_only': '#666666', # Medium gray
    'voice_preserving': '#999999',  # Light gray
}

MODEL_LABELS = {
    'gpt54': 'GPT',
    'claude_sonnet': 'Claude',
    'gemini_31_pro': 'Gemini',
}

# All 10 dimensions ordered by theoretical relevance for DH:
# Voice/authenticity first (core argument), then established DH markers,
# then exploratory dimensions (surface → deep → information-theoretic)
DIMENSION_MARKERS = {
    'Voice/Authenticity': ['voice_contraction_density', 'voice_first_person_density', 'voice_disfluency', 'voice_proper_noun_density', 'voice_specific_number_density'],
    'Lexical Diversity': ['ld_unique_words', 'ld_mtld', 'ld_hapax_ratio'],
    'Stylometric': ['pos_adj_ratio', 'pos_adv_ratio', 'pos_noun_ratio', 'pos_pron_ratio', 'pos_punct_ratio', 'sty_exclamation_per_sent', 'sty_function_word_ratio', 'sty_question_per_sent'],
    'Readability': ['read_flesch_ease', 'read_coleman_liau', 'read_syllables_per_word', 'sty_mean_word_length'],
    'Syntactic Complexity': ['syn_mean_sent_len', 'syn_std_sent_len', 'syn_n_sents', 'syn_subordination_ratio', 'syn_mean_depth', 'syn_min_sent_len'],
    'Semantic Coherence': ['coh_local_similarity', 'coh_global_similarity', 'sd_max_consecutive', 'sd_spread', 'sd_std_consecutive'],
    'Emotional Dynamics': ['emo_granularity', 'emo_volatility', 'emo_word_density'],
    'Sentiment': ['sent_mean_compound', 'sent_std_compound', 'sent_pct_positive', 'sent_pct_negative', 'sent_min_compound', 'sent_max_compound'],
    'Narrative Structure': ['narr_past_tense_ratio', 'narr_action_verb_density', 'narr_temporal_marker_density', 'causal_markers', 'moral_loyalty', 'conflict_second_person'],
    'Textual Entropy': ['ent_char', 'ent_word'],
}


def load_comparison(condition: str, model: str) -> pd.DataFrame:
    """Load a comparison CSV."""
    path = RESULTS_DIR / f"comparison_{condition}_{model}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_dimension_effects(df: pd.DataFrame) -> dict:
    """Compute mean effect size per dimension.

    Note: Cohen's d in the data is positive when original > rewrite.
    We flip the sign so positive = rewrite increases (inflation).
    """
    results = {}
    for dim, markers in DIMENSION_MARKERS.items():
        dim_data = df[df['marker'].isin(markers)]
        if len(dim_data) > 0:
            # Flip sign: positive d now means rewrite is HIGHER (inflation)
            mean_d = -dim_data['cohens_d'].mean()
            results[dim] = mean_d
        else:
            results[dim] = 0.0
    return results


def plot_narrative_normalization():
    """Create the three-panel figure."""
    fig = plt.figure(figsize=(16, 5), dpi=150)

    # Create grid: three panels side by side (increased wspace to prevent label overlap)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1], wspace=0.7)

    ax1 = fig.add_subplot(gs[0])  # Panel A: Heatmap
    ax2 = fig.add_subplot(gs[1])  # Panel B: Horizontal bars
    ax3 = fig.add_subplot(gs[2])  # Panel C: Contraction density

    models = ['gpt54', 'claude_sonnet', 'gemini_31_pro']
    dimensions = list(DIMENSION_MARKERS.keys())

    # ── Load data ──
    data = {}
    for model in models:
        df = load_comparison('generic', model)
        if df is not None:
            data[model] = df

    # ── Panel A: Heatmap ──
    effect_matrix = np.zeros((len(dimensions), len(models)))

    for j, model in enumerate(models):
        if model in data:
            effects = compute_dimension_effects(data[model])
            for i, dim in enumerate(dimensions):
                effect_matrix[i, j] = effects.get(dim, 0)

    # Create diverging colormap: blue (negative/deflation) - white - red (positive/inflation)
    cmap = LinearSegmentedColormap.from_list('diverging',
        ['#3274A1', '#FFFFFF', '#E1725B'], N=256)

    # Plot heatmap
    vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max()), 2.0)
    im = ax1.imshow(effect_matrix, cmap=cmap, aspect='auto',
                    vmin=-vmax, vmax=vmax)

    # Add text annotations
    for i in range(len(dimensions)):
        for j in range(len(models)):
            val = effect_matrix[i, j]
            color = 'white' if abs(val) > 0.8 else 'black'
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax1.set_yticks(range(len(dimensions)))
    ax1.set_yticklabels(dimensions)
    ax1.set_title('(A) Dimension Effects (Generic)', fontweight='bold', loc='left')

    # ── Panel B: Effect Sizes by Dimension Across All Conditions ──
    # Load all condition data
    all_cond_data = {'generic': data}
    for cond in ['rewrite_only', 'voice_preserving']:
        all_cond_data[cond] = {}
        for model in models:
            df = load_comparison(cond, model)
            if df is not None:
                all_cond_data[cond][model] = df

    # Compute mean absolute effect sizes across models for each dimension
    y_pos = np.arange(len(dimensions))
    width = 0.25
    conditions_b = ['generic', 'rewrite_only', 'voice_preserving']
    cond_colors = {'generic': '#333333', 'rewrite_only': '#666666', 'voice_preserving': '#999999'}
    cond_labels = {'generic': 'Generic', 'rewrite_only': 'Rewrite-only', 'voice_preserving': 'Voice-preserving'}

    for i, cond in enumerate(conditions_b):
        cond_means = []
        for dim in dimensions:
            markers = DIMENSION_MARKERS[dim]
            vals = []
            for model in models:
                if model in all_cond_data[cond]:
                    dim_data = all_cond_data[cond][model][all_cond_data[cond][model]['marker'].isin(markers)]
                    if len(dim_data) > 0:
                        vals.append(dim_data['cohens_d'].abs().mean())
            cond_means.append(np.mean(vals) if vals else 0)

        offset = (i - 1) * width
        ax2.barh(y_pos + offset, cond_means, width,
                 label=cond_labels[cond], color=cond_colors[cond],
                 edgecolor='#222222', linewidth=0.5)

    ax2.axvline(0, color='#222222', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(dimensions)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |Cohen's d|", fontweight='bold')
    ax2.set_title('(B) Mitigation by Dimension', fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=8)

    # ── Panel C: Key Markers Under Generic Prompt ──
    # Selected based on: largest effect sizes + theoretical importance for voice/style
    key_markers = [
        ('voice_contraction_density', 'Contractions'),
        ('ld_mtld', 'Lexical\nDiversity'),
        ('read_flesch_ease', 'Readability'),
        ('voice_first_person_density', 'First-Person'),
    ]

    x = np.arange(len(key_markers))
    width = 0.25
    model_colors = ['#333333', '#666666', '#999999']

    for i, model in enumerate(models):
        vals = []
        for marker_id, _ in key_markers:
            if model in data:
                row = data[model][data[model]['marker'] == marker_id]
                vals.append(row['pct_change'].iloc[0] if len(row) > 0 else 0)
            else:
                vals.append(0)

        offset = (i - 1) * width
        ax3.bar(x + offset, vals, width,
                color=model_colors[i],
                edgecolor='#222222', linewidth=0.5,
                label=MODEL_LABELS[model])

    ax3.axhline(0, color='#222222', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels([label.replace('\n', ' ') for _, label in key_markers],
                        rotation=45, ha='right')
    ax3.set_ylabel('Change (%)', fontweight='bold')
    ax3.set_title('(C) Key Markers (Generic Prompt)', fontweight='bold', loc='left')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    return fig


def main():
    print("Generating narrative normalization figure...")
    fig = plot_narrative_normalization()

    # Create directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to figures directory
    fig.savefig(FIGURES_DIR / "fig_narrative_normalization.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(FIGURES_DIR / "fig_narrative_normalization.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {FIGURES_DIR / 'fig_narrative_normalization.pdf'}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
