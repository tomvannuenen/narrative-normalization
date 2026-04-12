#!/usr/bin/env python3
"""Generate three-panel stylometric convergence figure.

Panel A: PCA projection showing originals scattered, rewrites clustered
Panel B: Distribution shift along PC1
Panel C: Attribution accuracy showing dramatic drop
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt

# Setup paths
FIGURES_DIR = PROJECT_ROOT / "figures"
PAPER_DIR = PROJECT_ROOT / "Narrative_Normalization" / "paper" / "dsh"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stylometric"

# Publication settings - larger fonts for full-width figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Grayscale colors
COLORS = {
    'original': '#AAAAAA',
    'gpt54': '#333333',
    'claude_sonnet': '#666666',
    'gemini_31_pro': '#999999',
}

MODEL_LABELS = {
    'gpt54': 'GPT-5.4',
    'claude_sonnet': 'Claude',
    'gemini_31_pro': 'Gemini',
}

# Core features for analysis
CORE_FEATURES = [
    'mfw_top10_coverage', 'mfw_top50_coverage', 'mfw_fw_ratio',
    'vocab_yules_k', 'vocab_honores_r',
    'char_3gram_entropy',
    'wlen_mean', 'wlen_std',
    'slen_mean', 'slen_std',
    'punct_comma_ratio', 'punct_dash_ratio',
]


def load_data(condition='generic'):
    """Load stylometric data."""
    orig_path = DATA_DIR / f"stylometric_original_{condition}.parquet"
    originals = pd.read_parquet(orig_path)

    models = ['gpt54', 'claude_sonnet', 'gemini_31_pro']
    rewrites_list = []

    for model in models:
        path = DATA_DIR / f"stylometric_{condition}_{model}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df['model'] = model
            rewrites_list.append(df)

    rewrites = pd.concat(rewrites_list, ignore_index=True)
    return originals, rewrites


def compute_attribution_accuracy(originals, rewrites, features):
    """Compute Delta-based attribution accuracy for each model."""
    orig_X = originals[features].values
    orig_X = np.nan_to_num(orig_X, nan=0.0)

    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(orig_X)

    results = {}
    models_list = ['gpt54', 'claude_sonnet', 'gemini_31_pro']

    for model in models_list:
        model_mask = rewrites['model'] == model
        model_rewrites = rewrites[model_mask].reset_index(drop=True)
        rewrite_X = model_rewrites[features].values
        rewrite_X = np.nan_to_num(rewrite_X, nan=0.0)
        rewrite_scaled = scaler.transform(rewrite_X)

        # For each original, find closest rewrite
        distances = cdist(orig_scaled, rewrite_scaled, metric='cityblock')
        closest_rewrite_idx = distances.argmin(axis=1)

        # Check if closest rewrite is the correct one (same id)
        orig_ids = originals['id'].values
        rewrite_ids = model_rewrites['id'].values
        correct = sum(orig_ids[i] == rewrite_ids[closest_rewrite_idx[i]]
                      for i in range(len(orig_ids)))

        accuracy = correct / len(originals) * 100
        results[model] = accuracy

    return results


def create_combined_figure():
    """Create three-panel figure: PCA + Distribution + Attribution."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Load data
    originals, rewrites = load_data('generic')

    # Get valid features
    valid_features = [f for f in CORE_FEATURES if f in originals.columns]

    # Prepare data for PCA
    orig_X = originals[valid_features].values
    rewrite_X = rewrites[valid_features].values
    orig_X = np.nan_to_num(orig_X, nan=0.0)
    rewrite_X = np.nan_to_num(rewrite_X, nan=0.0)

    X_all = np.vstack([orig_X, rewrite_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    orig_pca = X_pca[:len(originals)]
    rewrite_pca = X_pca[len(originals):]
    models = rewrites['model'].values

    # ═══════════════════════════════════════════════════════════════
    # Panel A: PCA Visualization
    # ═══════════════════════════════════════════════════════════════
    # Plot originals
    ax1.scatter(orig_pca[:, 0], orig_pca[:, 1],
               c=COLORS['original'], alpha=0.5, s=30,
               label='Original', marker='o', edgecolors='none')

    # Plot rewrites by model
    for model in ['gpt54', 'claude_sonnet', 'gemini_31_pro']:
        mask = models == model
        if mask.sum() > 0:
            ax1.scatter(rewrite_pca[mask, 0], rewrite_pca[mask, 1],
                       c=COLORS[model], alpha=0.6, s=35,
                       label=MODEL_LABELS[model], marker='s',
                       edgecolors='#222222', linewidths=0.4)

    # Centroids - LARGER and more visible
    orig_centroid = orig_pca.mean(axis=0)
    rewrite_centroid = rewrite_pca.mean(axis=0)

    ax1.scatter(*orig_centroid, c='white', s=200, marker='X',
               edgecolors='black', linewidths=3, zorder=10,
               label='Centroid (orig)')
    ax1.scatter(*rewrite_centroid, c='black', s=200, marker='X',
               edgecolors='black', linewidths=3, zorder=10,
               label='Centroid (rewrite)')

    # Arrow - THICKER
    ax1.annotate('', xy=rewrite_centroid, xytext=orig_centroid,
                arrowprops=dict(arrowstyle='->', color='black', lw=3))

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax1.set_title('(A) Stylometric Space', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ═══════════════════════════════════════════════════════════════
    # Panel B: PC1 Distribution Shift (Density Plot)
    # ═══════════════════════════════════════════════════════════════
    orig_pc1 = orig_pca[:, 0]
    x_range = np.linspace(-8, 10, 200)

    # Original distribution
    kde_orig = stats.gaussian_kde(orig_pc1)
    ax2.fill_between(x_range, kde_orig(x_range), alpha=0.3,
                    color=COLORS['original'], label='Original')
    ax2.plot(x_range, kde_orig(x_range), color=COLORS['original'],
            linewidth=2.5, linestyle='--')

    # Each model's distribution
    for model in ['gpt54', 'claude_sonnet', 'gemini_31_pro']:
        mask = models == model
        model_pc1 = rewrite_pca[mask, 0]
        kde_model = stats.gaussian_kde(model_pc1)
        ax2.plot(x_range, kde_model(x_range), color=COLORS[model],
                linewidth=2.5, label=MODEL_LABELS[model])

    # Add vertical lines for means
    ax2.axvline(orig_pc1.mean(), color=COLORS['original'],
               linestyle=':', linewidth=2, alpha=0.7)
    for model in ['gpt54', 'claude_sonnet', 'gemini_31_pro']:
        mask = models == model
        ax2.axvline(rewrite_pca[mask, 0].mean(), color=COLORS[model],
                   linestyle=':', linewidth=2, alpha=0.7)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('(B) Distribution Shift', fontweight='bold', loc='left')
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.set_xlim(-7, 9)
    ax2.set_ylim(0, None)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ═══════════════════════════════════════════════════════════════
    # Panel C: Attribution Accuracy
    # ═══════════════════════════════════════════════════════════════
    # Compute attribution accuracy
    accuracy = compute_attribution_accuracy(originals, rewrites, valid_features)

    model_names = ['GPT-5.4', 'Claude', 'Gemini']
    model_keys = ['gpt54', 'claude_sonnet', 'gemini_31_pro']
    acc_values = [accuracy[k] for k in model_keys]

    x_pos = np.arange(len(model_names))
    colors = [COLORS[k] for k in model_keys]

    bars = ax3.bar(x_pos, acc_values, color=colors,
                   edgecolor='#333333', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, acc_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Chance baseline
    chance = 100 / len(originals)  # 0.33%
    ax3.axhline(chance, color='#999999', linestyle='--', linewidth=2,
               label=f'Chance ({chance:.1f}%)')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names)
    ax3.set_ylabel('Attribution Accuracy (%)', fontweight='bold')
    ax3.set_title('(C) Authorship Attribution', fontweight='bold', loc='left')
    ax3.set_ylim(0, 22)
    ax3.legend(loc='upper right', framealpha=0.95)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def main():
    print("Generating stylometric convergence figure...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    fig = create_combined_figure()

    for output_dir in [FIGURES_DIR, PAPER_DIR]:
        for fmt in ['pdf', 'png']:
            fig.savefig(output_dir / f'fig_stylometric_convergence.{fmt}',
                       dpi=300, bbox_inches='tight')

    print(f"Saved: fig_stylometric_convergence.pdf")
    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
