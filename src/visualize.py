"""Visualisations for the AI text sanitization study.

Publication-quality figures following consistent styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

# ── Publication settings ────────────────────────────────────────────────
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
    'gradient': ['#dddddd', '#bbbbbb', '#888888', '#555555', '#222222'],
    'significant': '#222222',
    'not_significant': '#bbbbbb',
    'original': '#222222',
    'rewrite': '#888888',
    'increase': '#555555',
    'decrease': '#999999',
}


def _savefig(fig, name: str):
    """Save figure as both PDF and PNG."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = RESULTS_DIR / f"{name}.pdf"
    png_path = RESULTS_DIR / f"{name}.png"

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {png_path}")


def _style_axis(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """Apply consistent axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if title:
        ax.set_title(title, fontweight='bold', loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')


def plot_effect_sizes(comparison: pd.DataFrame, model_label: str = ""):
    """Horizontal bar chart of Cohen's d for each marker."""
    df = comparison.sort_values("cohens_d")
    sig = df["significant_fdr"] if "significant_fdr" in df.columns else df["significant"]

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.25)), dpi=150)

    # Color by significance
    colors = [COLORS['significant'] if s else COLORS['not_significant'] for s in sig]

    bars = ax.barh(
        df["marker"],
        df["cohens_d"],
        color=colors,
        edgecolor='#222222',
        linewidth=0.5,
    )

    ax.axvline(0, color="#222222", linewidth=1.2)

    # Add value labels for significant markers
    for bar, (_, row) in zip(bars, df.iterrows()):
        if row.get("significant_fdr", row.get("significant", False)):
            val = row["cohens_d"]
            offset = 0.05 if val >= 0 else -0.05
            ha = 'left' if val >= 0 else 'right'
            ax.annotate(
                f'{val:.2f}',
                xy=(val + offset, bar.get_y() + bar.get_height()/2),
                va='center', ha=ha, fontsize=7, color='#444444'
            )

    _style_axis(
        ax,
        title=f"Effect Sizes: Original vs. Rewrite — {model_label}" if model_label else "Effect Sizes",
        xlabel="Cohen's d (positive = original higher, negative = rewrite higher)",
    )

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['significant'], ec='#222222', lw=0.5, label="Significant (FDR)"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['not_significant'], ec='#222222', lw=0.5, label="Not significant"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _savefig(fig, f"effect_sizes_{model_label or 'all'}")


def plot_dimension_summary(summary: pd.DataFrame, model_label: str = ""):
    """Grouped bar chart showing n_significant and mean effect size per dimension."""
    # Sort by mean effect size
    summary = summary.sort_values("mean_abs_cohens_d", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # ── Left panel: number of significant markers ──
    ax = axes[0]
    bars = ax.barh(
        summary["dimension"],
        summary["n_significant"],
        color=COLORS['gradient'][2],
        edgecolor='#222222',
        linewidth=0.5,
    )

    # Add value labels
    for bar, val in zip(bars, summary["n_significant"]):
        ax.annotate(
            f'{int(val)}',
            xy=(val + 0.2, bar.get_y() + bar.get_height()/2),
            va='center', ha='left', fontsize=9, fontweight='bold'
        )

    _style_axis(
        ax,
        title="(A) Significant Changes by Dimension",
        xlabel="# Significant markers (FDR-corrected)",
    )
    ax.set_xlim(0, summary["n_significant"].max() * 1.2)

    # ── Right panel: mean |Cohen's d| ──
    ax = axes[1]
    bars = ax.barh(
        summary["dimension"],
        summary["mean_abs_cohens_d"],
        color=COLORS['gradient'][3],
        edgecolor='#222222',
        linewidth=0.5,
    )

    # Add value labels
    for bar, val in zip(bars, summary["mean_abs_cohens_d"]):
        ax.annotate(
            f'{val:.2f}',
            xy=(val + 0.02, bar.get_y() + bar.get_height()/2),
            va='center', ha='left', fontsize=9, fontweight='bold'
        )

    _style_axis(
        ax,
        title="(B) Effect Size by Dimension",
        xlabel="Mean |Cohen's d|",
    )
    ax.set_xlim(0, summary["mean_abs_cohens_d"].max() * 1.2)

    fig.suptitle(
        f"Dimension-Level Summary — {model_label}" if model_label else "Dimension-Level Summary",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, f"dimension_summary_{model_label or 'all'}")


def plot_paired_distributions(
    orig_markers: pd.DataFrame,
    rewrite_markers: pd.DataFrame,
    cols: list[str],
    model_label: str = "",
):
    """Overlay histograms of selected markers for original vs. rewrite."""
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=150)
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        orig_data = orig_markers[col].dropna()
        rewrite_data = rewrite_markers[col].dropna()

        # Use histograms with transparency for better comparison
        bins = np.histogram_bin_edges(
            np.concatenate([orig_data, rewrite_data]), bins=30
        )

        ax.hist(
            orig_data, bins=bins, alpha=0.7,
            color=COLORS['original'], edgecolor='#222222', linewidth=0.5,
            label='Original', density=True
        )
        ax.hist(
            rewrite_data, bins=bins, alpha=0.5,
            color=COLORS['rewrite'], edgecolor='#222222', linewidth=0.5,
            label='Rewrite', density=True
        )

        # Add means as vertical lines
        ax.axvline(orig_data.mean(), color=COLORS['original'], linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(rewrite_data.mean(), color=COLORS['rewrite'], linestyle='--', linewidth=1.5, alpha=0.8)

        _style_axis(ax, title=col)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_ylabel('Density', fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Distribution Comparison — {model_label}" if model_label else "Distribution Comparison",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, f"distributions_{model_label or 'all'}")


def plot_cross_model_comparison(
    comparisons: dict[str, pd.DataFrame],
    top_n: int = 20,
):
    """Compare effect sizes across models for top markers."""
    # Get all markers and their mean absolute effect across models
    all_markers = set()
    for df in comparisons.values():
        all_markers.update(df["marker"].tolist())

    # Calculate mean absolute d across models
    marker_importance = {}
    for marker in all_markers:
        ds = []
        for df in comparisons.values():
            row = df[df["marker"] == marker]
            if len(row) > 0:
                ds.append(abs(row["cohens_d"].iloc[0]))
        marker_importance[marker] = np.mean(ds) if ds else 0

    # Select top N markers
    top_markers = sorted(marker_importance.keys(), key=lambda x: marker_importance[x], reverse=True)[:top_n]

    # Prepare data
    models = list(comparisons.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)), dpi=150)

    bar_height = 0.8 / n_models
    y_positions = np.arange(len(top_markers))

    for i, (model_label, df) in enumerate(comparisons.items()):
        ds = []
        for marker in top_markers:
            row = df[df["marker"] == marker]
            ds.append(row["cohens_d"].iloc[0] if len(row) > 0 else 0)

        offset = (i - n_models/2 + 0.5) * bar_height
        ax.barh(
            y_positions + offset, ds,
            height=bar_height,
            color=COLORS['gradient'][i + 1],
            edgecolor='#222222',
            linewidth=0.5,
            label=model_label,
        )

    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_markers)
    ax.invert_yaxis()

    _style_axis(
        ax,
        title=f"Cross-Model Effect Size Comparison (Top {top_n} Markers)",
        xlabel="Cohen's d",
    )
    ax.legend(loc='lower right', framealpha=0.9)

    fig.tight_layout()
    _savefig(fig, "cross_model_comparison")


def plot_transformation_summary(
    trans_dfs: dict[str, pd.DataFrame],
):
    """Bar chart comparing transformation metrics across models."""
    metrics = [
        ('trans_char_edit_dist_norm', 'Edit Distance (norm)'),
        ('trans_word_jaccard', 'Word Jaccard'),
        ('trans_semantic_sim', 'Semantic Similarity'),
        ('trans_length_ratio', 'Length Ratio'),
    ]

    models = list(trans_dfs.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), dpi=150)

    for ax, (metric, label) in zip(axes, metrics):
        means = []
        stds = []
        for model in models:
            df = trans_dfs[model]
            if metric in df.columns:
                means.append(df[metric].mean())
                stds.append(df[metric].std() / np.sqrt(len(df)))  # SEM
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(n_models)
        bars = ax.bar(
            x, means, yerr=stds,
            color=[COLORS['gradient'][i + 1] for i in range(n_models)],
            edgecolor='#222222',
            linewidth=0.5,
            capsize=4,
            error_kw={'lw': 1.2},
        )

        # Add value labels
        for bar, val in zip(bars, means):
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        _style_axis(ax, title=label)

    fig.suptitle(
        "Transformation Metrics by Model",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, "transformation_summary")
