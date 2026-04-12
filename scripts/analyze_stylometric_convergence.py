#!/usr/bin/env python3
"""Analyze stylometric convergence: Delta analysis and PCA visualization.

This script implements two analyses for the DSH paper:

1. Burrows' Delta Analysis:
   - Compute pairwise Delta distances within originals vs within rewrites
   - Show that rewrites cluster more tightly (lower intra-group distance)
   - This demonstrates convergence toward a shared stylometric profile

2. PCA Visualization:
   - Project texts into 2D stylometric space using MFW features
   - Show originals scattered, rewrites clustered
   - Visual evidence of normalization

Usage:
    python scripts/analyze_stylometric_convergence.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import combinations

# Setup paths
FIGURES_DIR = project_root / "figures"
PAPER_DIR = project_root / "Narrative_Normalization" / "paper" / "dsh"

# Publication settings (matching other paper figures)
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

# Grayscale color palette
COLORS = {
    'original': '#AAAAAA',      # Light gray for originals
    'gpt54': '#333333',         # Dark gray
    'claude_sonnet': '#666666', # Medium gray
    'gemini_31_pro': '#999999', # Light gray
}

MODEL_LABELS = {
    'gpt54': 'GPT',
    'claude_sonnet': 'Claude',
    'gemini_31_pro': 'Gemini',
}


# Aggregate MFW features (Delta-style)
MFW_AGGREGATE_FEATURES = [
    'mfw_top10_coverage', 'mfw_top50_coverage', 'mfw_concentration', 'mfw_fw_ratio',
    'delta_fw_zscore_mean', 'delta_fw_zscore_std', 'delta_fw_deviation',
]

# Vocabulary richness features
VOCAB_FEATURES = [
    'vocab_yules_k', 'vocab_simpsons_d', 'vocab_honores_r',
    'vocab_sichels_s', 'vocab_brunets_w',
]

# Character n-gram features
CHAR_NGRAM_FEATURES = [
    'char_2gram_entropy', 'char_3gram_entropy', 'char_4gram_entropy',
    'char_2gram_hapax_ratio', 'char_3gram_hapax_ratio', 'char_4gram_hapax_ratio',
]

# Word length features
WLEN_FEATURES = [
    'wlen_mean', 'wlen_std', 'wlen_skew',
    'wlen_1', 'wlen_2', 'wlen_3', 'wlen_4', 'wlen_5',
]

# Sentence length features
SLEN_FEATURES = [
    'slen_mean', 'slen_std', 'slen_variation_coef',
    'slen_short_ratio', 'slen_long_ratio',
]

# Punctuation features
PUNCT_FEATURES = [
    'punct_comma_ratio', 'punct_dash_ratio', 'punct_semicolon_ratio',
    'punct_colon_ratio', 'punct_exclamation_ratio', 'punct_question_ratio',
]

# Full stylometric feature set
STYLOMETRIC_FEATURES = (
    MFW_AGGREGATE_FEATURES + VOCAB_FEATURES + CHAR_NGRAM_FEATURES +
    WLEN_FEATURES + SLEN_FEATURES + PUNCT_FEATURES
)

# Core features most sensitive to authorship (for PCA)
CORE_STYLOMETRIC_FEATURES = [
    'mfw_top10_coverage', 'mfw_top50_coverage', 'mfw_fw_ratio',
    'vocab_yules_k', 'vocab_honores_r',
    'char_3gram_entropy',
    'wlen_mean', 'wlen_std',
    'slen_mean', 'slen_std',
    'punct_comma_ratio', 'punct_dash_ratio',
]


def load_stylometric_data(condition: str = "generic"):
    """Load stylometric data for a given condition."""
    data_dir = project_root / "data" / "processed" / "stylometric"

    # Load originals
    orig_path = data_dir / f"stylometric_original_{condition}.parquet"
    if not orig_path.exists():
        print(f"  Original data not found: {orig_path}")
        return None, None

    originals = pd.read_parquet(orig_path)
    print(f"  Loaded {len(originals)} original texts")

    # Load rewrites for each model
    models = ["gpt54", "claude_sonnet", "gemini_31_pro"]
    rewrites_list = []

    for model in models:
        path = data_dir / f"stylometric_{condition}_{model}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df['model'] = model
            rewrites_list.append(df)
            print(f"  Loaded {len(df)} {model} rewrites")

    if not rewrites_list:
        return originals, None

    rewrites = pd.concat(rewrites_list, ignore_index=True)
    return originals, rewrites


def compute_burrows_delta(texts_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Compute pairwise Burrows' Delta distances.

    Delta = mean(|z_i - z_j|) across all features,
    where z-scores are computed across the corpus.
    """
    # Extract feature matrix
    X = texts_df[feature_cols].values

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    # Z-score normalize (corpus-level, following Burrows)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    # Compute pairwise Manhattan distances (Delta = mean absolute z-score diff)
    # pdist computes condensed distance matrix
    distances = pdist(X_z, metric='cityblock') / len(feature_cols)

    return distances


def analyze_convergence(originals: pd.DataFrame, rewrites: pd.DataFrame,
                       feature_cols: list[str], feature_name: str = "MFW"):
    """Analyze stylometric convergence using Delta distances."""
    print(f"\n{'='*60}")
    print(f"BURROWS' DELTA ANALYSIS ({feature_name} features)")
    print(f"{'='*60}")

    # Filter to valid features
    valid_features = [f for f in feature_cols if f in originals.columns and f in rewrites.columns]
    print(f"Using {len(valid_features)} features")

    if len(valid_features) < 5:
        print("  Not enough features available")
        return None

    # Compute intra-group distances for originals
    orig_distances = compute_burrows_delta(originals, valid_features)
    orig_mean = np.mean(orig_distances)
    orig_std = np.std(orig_distances)

    print(f"\nOriginal texts:")
    print(f"  Mean Delta distance: {orig_mean:.4f} (SD = {orig_std:.4f})")
    print(f"  Range: [{np.min(orig_distances):.4f}, {np.max(orig_distances):.4f}]")

    # Compute intra-group distances for rewrites (all models combined)
    rewrite_distances = compute_burrows_delta(rewrites, valid_features)
    rewrite_mean = np.mean(rewrite_distances)
    rewrite_std = np.std(rewrite_distances)

    print(f"\nRewritten texts (all models):")
    print(f"  Mean Delta distance: {rewrite_mean:.4f} (SD = {rewrite_std:.4f})")
    print(f"  Range: [{np.min(rewrite_distances):.4f}, {np.max(rewrite_distances):.4f}]")

    # Convergence ratio
    convergence = orig_mean / rewrite_mean if rewrite_mean > 0 else float('inf')
    print(f"\nConvergence ratio (orig/rewrite): {convergence:.2f}x")

    if convergence > 1:
        print(f"  -> Rewrites are {(convergence-1)*100:.1f}% MORE similar to each other than originals")
    else:
        print(f"  -> Rewrites are {(1-convergence)*100:.1f}% LESS similar to each other than originals")

    # Per-model analysis
    print(f"\nPer-model analysis:")
    models = rewrites['model'].unique()
    model_results = {}

    for model in models:
        model_df = rewrites[rewrites['model'] == model]
        model_distances = compute_burrows_delta(model_df, valid_features)
        model_mean = np.mean(model_distances)
        model_results[model] = model_mean
        print(f"  {model}: Mean Delta = {model_mean:.4f}")

    return {
        'orig_mean': orig_mean,
        'orig_std': orig_std,
        'rewrite_mean': rewrite_mean,
        'rewrite_std': rewrite_std,
        'convergence_ratio': convergence,
        'model_results': model_results,
    }


def create_pca_visualization(originals: pd.DataFrame, rewrites: pd.DataFrame,
                            feature_cols: list[str], output_path: Path,
                            title: str = "Stylometric Space"):
    """Create PCA visualization of stylometric space (publication style)."""
    print(f"\n{'='*60}")
    print(f"PCA VISUALIZATION")
    print(f"{'='*60}")

    # Filter to valid features
    valid_features = [f for f in feature_cols if f in originals.columns and f in rewrites.columns]
    print(f"Using {len(valid_features)} features for PCA")

    # Prepare data
    orig_X = originals[valid_features].values
    rewrite_X = rewrites[valid_features].values

    # Handle missing values
    orig_X = np.nan_to_num(orig_X, nan=0.0)
    rewrite_X = np.nan_to_num(rewrite_X, nan=0.0)

    # Combine and standardize
    X_all = np.vstack([orig_X, rewrite_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Split back
    orig_pca = X_pca[:len(originals)]
    rewrite_pca = X_pca[len(originals):]

    # Get model labels for rewrites
    models = rewrites['model'].values

    # Create figure (publication size)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot originals (light gray circles)
    ax.scatter(orig_pca[:, 0], orig_pca[:, 1],
               c=COLORS['original'], alpha=0.5, s=25,
               label='Original', marker='o', edgecolors='none')

    # Plot rewrites by model (grayscale squares)
    for model in ['gpt54', 'claude_sonnet', 'gemini_31_pro']:
        mask = models == model
        if mask.sum() > 0:
            ax.scatter(rewrite_pca[mask, 0], rewrite_pca[mask, 1],
                      c=COLORS[model], alpha=0.6, s=35,
                      label=MODEL_LABELS.get(model, model), marker='s',
                      edgecolors='#222222', linewidths=0.3)

    # Compute and show centroids
    orig_centroid = orig_pca.mean(axis=0)
    ax.scatter(*orig_centroid, c='white', s=150, marker='X',
               edgecolors='#222222', linewidths=2, label='Original centroid', zorder=10)

    rewrite_centroid = rewrite_pca.mean(axis=0)
    ax.scatter(*rewrite_centroid, c='#333333', s=150, marker='X',
               edgecolors='#222222', linewidths=2, label='Rewrite centroid', zorder=10)

    # Draw arrow from original to rewrite centroid
    ax.annotate('', xy=rewrite_centroid, xytext=orig_centroid,
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
    ax.set_title(title, fontweight='bold', loc='left')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save to figures directory
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")

    # Save copy to paper directory
    paper_pdf = PAPER_DIR / output_path.with_suffix('.pdf').name.replace('pca_stylometric_', 'fig_stylometric_')
    plt.savefig(paper_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved to paper dir: {paper_pdf}")

    # Compute spread statistics
    orig_spread = np.std(orig_pca, axis=0).mean()
    rewrite_spread = np.std(rewrite_pca, axis=0).mean()

    print(f"\nSpread in PCA space:")
    print(f"  Originals: {orig_spread:.4f}")
    print(f"  Rewrites:  {rewrite_spread:.4f}")
    print(f"  Reduction: {(1 - rewrite_spread/orig_spread)*100:.1f}%")

    return pca, orig_pca, rewrite_pca


def compute_cross_distances(originals: pd.DataFrame, rewrites: pd.DataFrame,
                           feature_cols: list[str]):
    """Compute distances between originals and their rewrites.

    This tests whether we can still match originals to their rewrites.
    """
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    # Filter to valid features
    valid_features = [f for f in feature_cols if f in originals.columns and f in rewrites.columns]

    # Ensure we have matching IDs
    common_ids = set(originals['id'].values) & set(rewrites['id'].values)
    print(f"Analyzing {len(common_ids)} text pairs")

    # Filter to common IDs
    orig_df = originals[originals['id'].isin(common_ids)].set_index('id')

    # For each model, compute matching accuracy
    models = rewrites['model'].unique()

    for model in models:
        model_rewrites = rewrites[rewrites['model'] == model]
        model_rewrites = model_rewrites[model_rewrites['id'].isin(common_ids)].set_index('id')

        # Align by ID
        common = orig_df.index.intersection(model_rewrites.index)
        if len(common) == 0:
            continue

        orig_aligned = orig_df.loc[common][valid_features].values
        rewrite_aligned = model_rewrites.loc[common][valid_features].values

        # Handle NaN
        orig_aligned = np.nan_to_num(orig_aligned, nan=0.0)
        rewrite_aligned = np.nan_to_num(rewrite_aligned, nan=0.0)

        # Stack and standardize
        all_X = np.vstack([orig_aligned, rewrite_aligned])
        scaler = StandardScaler()
        all_X_z = scaler.fit_transform(all_X)

        orig_z = all_X_z[:len(common)]
        rewrite_z = all_X_z[len(common):]

        # For each original, find closest rewrite
        correct = 0
        for i in range(len(common)):
            distances = np.abs(orig_z[i] - rewrite_z).mean(axis=1)
            closest = np.argmin(distances)
            if closest == i:
                correct += 1

        accuracy = correct / len(common)
        print(f"\n{model}:")
        print(f"  Attribution accuracy: {accuracy*100:.1f}%")
        print(f"  (Chance = {100/len(common):.1f}%)")


def main():
    print("="*60)
    print("STYLOMETRIC CONVERGENCE ANALYSIS")
    print("="*60)

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze each condition
    conditions = ["generic", "voice_preserving", "rewrite_only"]

    all_results = {}

    for condition in conditions:
        print(f"\n\n{'#'*60}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'#'*60}")

        originals, rewrites = load_stylometric_data(condition)

        if originals is None or rewrites is None:
            print("  Skipping - data not available")
            continue

        # 1. Delta analysis with core stylometric features
        core_results = analyze_convergence(
            originals, rewrites,
            CORE_STYLOMETRIC_FEATURES,
            feature_name="Core stylometric (12 features)"
        )

        # 2. Delta analysis with full feature set
        full_results = analyze_convergence(
            originals, rewrites,
            STYLOMETRIC_FEATURES,
            feature_name="Full stylometric (all features)"
        )

        # 3. PCA visualization (core features)
        pca, orig_pca, rewrite_pca = create_pca_visualization(
            originals, rewrites,
            CORE_STYLOMETRIC_FEATURES,
            FIGURES_DIR / f"pca_stylometric_{condition}.png",
            title=f"Stylometric Space ({condition.replace('_', ' ').title()})"
        )

        # 4. Attribution analysis
        compute_cross_distances(originals, rewrites, CORE_STYLOMETRIC_FEATURES)

        all_results[condition] = {
            'core': core_results,
            'full': full_results,
        }

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY ACROSS CONDITIONS")
    print("="*60)

    for condition, results in all_results.items():
        if results['core']:
            conv = results['core']['convergence_ratio']
            print(f"\n{condition}:")
            print(f"  Stylometric convergence ratio: {conv:.2f}x")
            if conv > 1:
                print(f"  Rewrites are {(conv-1)*100:.1f}% MORE similar to each other")
            else:
                print(f"  Rewrites are {(1-conv)*100:.1f}% LESS similar to each other")

    print(f"\n\nOutput saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
