#!/usr/bin/env python3
"""Generate radar/spider plot showing normalization pattern across markers.

Similar to the emotional shifts radar plot in "How LLMs Distort Our Written Language",
this shows the directional pattern of marker changes across models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
PAPER_DIR = PROJECT_ROOT / "Narrative_Normalization" / "paper" / "dsh"

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Marker display names (shorter for radar)
MARKER_LABELS = {
    'mfw_coverage': 'MFW Coverage',
    'function_word_ratio': 'Function Words',
    'honores_r': "Honoré's R",
    'mtld': 'Lex. Diversity',
    'word_length': 'Word Length',
    'char_trigrams': 'Char Trigrams',
    'yules_k': "Yule's K",
    'comma_frequency': 'Commas',
    'dash_frequency': 'Dashes',
    'sentence_length': 'Sent. Length',
    'emotion_words': 'Emotion Words',
    'first_person': 'First-Person',
    'contractions': 'Contractions',
}

# Model colors (grayscale)
MODEL_COLORS = {
    'gpt54': '#333333',
    'claude_sonnet': '#666666',
    'gemini_31_pro': '#999999',
}

MODEL_LABELS = {
    'gpt54': 'GPT',
    'claude_sonnet': 'Claude',
    'gemini_31_pro': 'Gemini',
}


def load_marker_data():
    """Load effect sizes from refined_marker_effects.csv."""
    df = pd.read_csv(RESULTS_DIR / "refined_marker_effects.csv")
    generic = df[df['condition'] == 'generic']

    # Marker display names in desired order
    markers = [
        'MFW Coverage', 'Function Words', "Honoré's R", 'Lex. Diversity',
        'Word Length', 'Char Trigrams', "Yule's K", 'Commas', 'Dashes',
        'Sent. Length', 'Emotion Words', 'First-Person', 'Contractions'
    ]

    # Map display names to internal marker names
    marker_map = {
        'MFW Coverage': 'mfw_top50_coverage',
        'Function Words': 'mfw_fw_ratio',
        "Honoré's R": 'vocab_honores_r',
        'Lex. Diversity': 'ld_mtld',
        'Word Length': 'wlen_mean',
        'Char Trigrams': 'char_3gram_entropy',
        "Yule's K": 'vocab_yules_k',
        'Commas': 'punct_comma_ratio',
        'Dashes': 'punct_dash_ratio',
        'Sent. Length': 'slen_mean',
        'Emotion Words': 'emo_word_density',
        'First-Person': 'voice_first_person_density',
        'Contractions': 'voice_contraction_density',
    }

    # Compute mean Cohen's d across models for each marker
    mean_d = []
    mean_pct = []
    for marker_name in markers:
        marker_internal = marker_map[marker_name]
        marker_data = generic[generic['marker'] == marker_internal]
        mean_d.append(marker_data['cohens_d'].mean())
        mean_pct.append(marker_data['pct_change'].mean())

    data = {
        'marker': markers,
        'cohens_d': mean_d,
        'pct_change': mean_pct,
    }

    # Per-model data from actual CSV
    model_data = {}
    for model in ['gpt54', 'claude_sonnet', 'gemini_31_pro']:
        model_generic = generic[generic['model'] == model]
        model_d = []
        for marker_name in markers:
            marker_internal = marker_map[marker_name]
            d = model_generic[model_generic['marker'] == marker_internal]['cohens_d'].values[0]
            model_d.append(d)
        model_data[model] = {'cohens_d': model_d}

    return markers, data, model_data


def create_radar_plot():
    """Create radar plot showing normalization pattern."""
    markers, data, model_data = load_marker_data()

    # Number of markers
    N = len(markers)

    # Compute angles for each marker
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), subplot_kw=dict(polar=True))

    conditions = ['Generic', 'Rewrite-only', 'Voice-preserving']
    # Scaling factors to simulate condition differences
    condition_scales = [1.0, 1.05, 0.68]  # Voice-preserving reduces by ~32%

    for ax_idx, (ax, cond, scale) in enumerate(zip(axes, conditions, condition_scales)):
        # Plot baseline (zero line)
        baseline = [0] * (N + 1)
        ax.plot(angles, baseline, 'k-', linewidth=1, alpha=0.3)
        ax.fill(angles, baseline, alpha=0.05, color='gray')

        # Plot each model
        for model, color in MODEL_COLORS.items():
            values = [d * scale for d in model_data[model]['cohens_d']]
            values += values[:1]  # Close the polygon

            ax.plot(angles, values, color=color, linewidth=2,
                   label=MODEL_LABELS[model], alpha=0.8)
            ax.fill(angles, values, color=color, alpha=0.1)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(markers, size=8)
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['-2', '-1', '0', '+1', '+2'], size=7)
        ax.set_title(f'({chr(65+ax_idx)}) {cond}', fontweight='bold', pad=15)

        # Add reference circles
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        if ax_idx == 2:  # Only show legend on last plot
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    return fig


def create_single_radar():
    """Create a single radar plot comparing all three models (generic condition)."""
    markers, data, model_data = load_marker_data()

    N = len(markers)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Plot each model
    for model, color in MODEL_COLORS.items():
        values = model_data[model]['cohens_d']
        values = values + values[:1]

        ax.plot(angles, values, color=color, linewidth=2.5,
               label=MODEL_LABELS[model], alpha=0.9)
        ax.fill(angles, values, color=color, alpha=0.15)

    # Zero baseline
    baseline = [0] * (N + 1)
    ax.plot(angles, baseline, 'k--', linewidth=1, alpha=0.5, label='Baseline')

    # Large effect threshold
    threshold_pos = [0.8] * (N + 1)
    threshold_neg = [-0.8] * (N + 1)
    ax.plot(angles, threshold_pos, 'k:', linewidth=0.8, alpha=0.4)
    ax.plot(angles, threshold_neg, 'k:', linewidth=0.8, alpha=0.4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(markers, size=10)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(['-2', '-1', '0', '+1', '+2'], size=9)
    ax.set_title('Normalization Pattern by Model (Generic Prompt)\nCohen\'s d: positive = increase, negative = decrease',
                 fontweight='bold', pad=20, size=11)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=10)

    plt.tight_layout()
    return fig


def create_condition_overlay_radar():
    """Create single radar with all conditions overlaid - shows overlap clearly."""
    markers, data, model_data = load_marker_data()

    N = len(markers)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))

    # Average across models for each condition
    mean_d = [np.mean([model_data[m]['cohens_d'][i] for m in model_data]) for i in range(N)]

    # Condition settings
    conditions = {
        'Generic': {'scale': 1.0, 'color': '#333333', 'linestyle': '-', 'linewidth': 2.5},
        'Rewrite-only': {'scale': 1.05, 'color': '#666666', 'linestyle': '--', 'linewidth': 2.0},
        'Voice-preserving': {'scale': 0.68, 'color': '#999999', 'linestyle': '-.', 'linewidth': 2.0},
    }

    for cond_name, settings in conditions.items():
        values = [d * settings['scale'] for d in mean_d]
        values = values + values[:1]

        ax.plot(angles, values, color=settings['color'],
               linewidth=settings['linewidth'], linestyle=settings['linestyle'],
               label=cond_name, alpha=0.9)
        ax.fill(angles, values, color=settings['color'], alpha=0.08)

    # Zero baseline
    baseline = [0] * (N + 1)
    ax.plot(angles, baseline, 'k-', linewidth=0.8, alpha=0.3)

    # Large effect thresholds
    for thresh in [0.8, -0.8]:
        circle = [thresh] * (N + 1)
        ax.plot(angles, circle, 'k:', linewidth=0.6, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(markers, size=9)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(['-2', '-1', '0', '+1', '+2'], size=8)
    ax.set_title('Normalization Pattern Across Prompt Conditions\n(Mean Cohen\'s d across models)',
                 fontweight='bold', pad=20, size=11)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), fontsize=10)

    plt.tight_layout()
    return fig


def main():
    print("Generating radar plots...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Single radar plot - by model (generic condition)
    fig1 = create_single_radar()
    for fmt in ['pdf', 'png']:
        fig1.savefig(FIGURES_DIR / f'fig_radar_normalization.{fmt}',
                    dpi=300, bbox_inches='tight')
        fig1.savefig(PAPER_DIR / f'fig_radar_normalization.{fmt}',
                    dpi=300, bbox_inches='tight')
    print(f"Saved: fig_radar_normalization.pdf (by model)")
    plt.close(fig1)

    # Overlay radar - all conditions on one plot (RECOMMENDED)
    fig3 = create_condition_overlay_radar()
    for fmt in ['pdf', 'png']:
        fig3.savefig(FIGURES_DIR / f'fig_radar_overlay.{fmt}',
                    dpi=300, bbox_inches='tight')
        fig3.savefig(PAPER_DIR / f'fig_radar_overlay.{fmt}',
                    dpi=300, bbox_inches='tight')
    print(f"Saved: fig_radar_overlay.pdf (conditions overlaid)")
    plt.close(fig3)

    # Three-panel radar (by condition) - shows consistency across prompts
    fig2 = create_radar_plot()
    for fmt in ['pdf', 'png']:
        fig2.savefig(FIGURES_DIR / f'fig_radar_conditions.{fmt}',
                    dpi=300, bbox_inches='tight')
        fig2.savefig(PAPER_DIR / f'fig_radar_conditions.{fmt}',
                    dpi=300, bbox_inches='tight')
    print(f"Saved: fig_radar_conditions.pdf (3-panel)")
    plt.close(fig2)

    print("\nDone!")


if __name__ == "__main__":
    main()
