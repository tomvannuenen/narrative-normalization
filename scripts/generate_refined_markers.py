#!/usr/bin/env python3
"""Generate the 13-marker refined analysis for the paper.

This script creates refined_marker_effects.csv by:
1. Computing stylometric markers from parquet files (pooled Cohen's d)
2. Extracting empathic/register markers from comparison CSV files (paired Cohen's d)

The 13 markers are grounded in computational stylistics and register studies:
- Function Words (2): MFW coverage, Function word ratio
- Vocabulary (5): MTLD, Honoré's R, Yule's K, Mean word length, Char trigram entropy
- Syntax & Punctuation (3): Mean sentence length, Comma frequency, Dash frequency
- Register (3): Contraction density, First-person pronoun density, Emotion word density
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
STYLOMETRIC_DIR = DATA_DIR / "stylometric"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS = ['gpt54', 'claude_sonnet', 'gemini_31_pro']
CONDITIONS = ['generic', 'rewrite_only', 'voice_preserving']

# The 13 refined markers with their categories, citations, and source
# source='stylometric': compute from stylometric parquet files
# source='comparison': extract from comparison CSV files
REFINED_MARKERS = {
    # Function Words (from stylometric parquets)
    'mfw_top50_coverage': {
        'category': 'Function Words',
        'citation': 'Burrows 2002; Eder et al. 2016',
        'source': 'stylometric',
    },
    'mfw_fw_ratio': {
        'category': 'Function Words',
        'citation': 'Burrows 2002; Eder et al. 2016',
        'source': 'stylometric',
    },
    # Vocabulary (mixed)
    'vocab_honores_r': {
        'category': 'Vocabulary',
        'citation': 'Honoré 1979',
        'source': 'stylometric',
    },
    'ld_mtld': {
        'category': 'Vocabulary',
        'citation': 'McCarthy & Jarvis 2010',
        'source': 'comparison',  # from comparison files
    },
    'wlen_mean': {
        'category': 'Vocabulary',
        'citation': 'Mendenhall 1887',
        'source': 'stylometric',
    },
    'char_3gram_entropy': {
        'category': 'Vocabulary',
        'citation': 'Stamatatos 2009, 2013',
        'source': 'stylometric',
    },
    'vocab_yules_k': {
        'category': 'Vocabulary',
        'citation': 'Yule 1944; Tweedie & Baayen 1998',
        'source': 'stylometric',
    },
    # Syntax & Punctuation (from stylometric parquets)
    'slen_mean': {
        'category': 'Syntax & Punctuation',
        'citation': 'Williams 1940; Stamatatos 2009',
        'source': 'stylometric',
    },
    'punct_comma_ratio': {
        'category': 'Syntax & Punctuation',
        'citation': 'Stamatatos 2009; Grieve 2007',
        'source': 'stylometric',
    },
    'punct_dash_ratio': {
        'category': 'Syntax & Punctuation',
        'citation': 'Stamatatos 2009',
        'source': 'stylometric',
    },
    # Register (from comparison files - uses paired Cohen's d)
    'voice_contraction_density': {
        'category': 'Register',
        'citation': 'Biber 1988; Heylighen & Dewaele 2002',
        'source': 'comparison',
    },
    'voice_first_person_density': {
        'category': 'Register',
        'citation': 'Biber 1988',
        'source': 'comparison',
    },
    'emo_word_density': {
        'category': 'Register',
        'citation': 'Pennebaker et al. 2015',
        'source': 'comparison',
    },
}


def cohens_d_pooled(orig, rewrite):
    """Compute Cohen's d with pooled standard deviation.

    Sign convention: positive = rewrite increases feature
    """
    n_orig, n_rewrite = len(orig), len(rewrite)
    if n_orig < 2 or n_rewrite < 2:
        return 0.0

    pooled_std = np.sqrt(
        ((n_orig - 1) * np.std(orig, ddof=1)**2 +
         (n_rewrite - 1) * np.std(rewrite, ddof=1)**2) /
        (n_orig + n_rewrite - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (np.mean(rewrite) - np.mean(orig)) / pooled_std


def load_stylometric(condition: str, model: str) -> tuple:
    """Load stylometric parquet files for orig and rewrite."""
    orig_path = STYLOMETRIC_DIR / f"stylometric_original_{condition}.parquet"
    rewrite_path = STYLOMETRIC_DIR / f"stylometric_{condition}_{model}.parquet"

    if not orig_path.exists() or not rewrite_path.exists():
        raise FileNotFoundError(f"Missing stylometric files for {condition}/{model}")

    return pd.read_parquet(orig_path), pd.read_parquet(rewrite_path)


def load_comparison(condition: str, model: str) -> pd.DataFrame:
    """Load comparison CSV file."""
    path = RESULTS_DIR / f"comparison_{condition}_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def compute_effects():
    """Compute effect sizes for all 13 markers."""
    rows = []

    for condition in CONDITIONS:
        for model in MODELS:
            # Load data
            try:
                sty_orig, sty_rewrite = load_stylometric(condition, model)
                comparison = load_comparison(condition, model)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

            for marker, info in REFINED_MARKERS.items():
                source = info['source']

                if source == 'stylometric':
                    # Compute from stylometric parquet files
                    if marker not in sty_orig.columns or marker not in sty_rewrite.columns:
                        print(f"Warning: {marker} not in stylometric data")
                        continue

                    orig_values = sty_orig[marker].dropna().values
                    rewrite_values = sty_rewrite[marker].dropna().values

                    n = min(len(orig_values), len(rewrite_values))
                    orig_values = orig_values[:n]
                    rewrite_values = rewrite_values[:n]

                    orig_mean = np.mean(orig_values)
                    rewrite_mean = np.mean(rewrite_values)
                    d = cohens_d_pooled(orig_values, rewrite_values)

                    try:
                        _, p_value = stats.wilcoxon(orig_values, rewrite_values)
                    except ValueError:
                        p_value = 1.0

                    pct_change = ((rewrite_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0

                else:  # source == 'comparison'
                    # Extract from comparison CSV file
                    marker_row = comparison[comparison['marker'] == marker]
                    if marker_row.empty:
                        print(f"Warning: {marker} not in comparison file")
                        continue

                    row = marker_row.iloc[0]
                    orig_mean = row['mean_orig']
                    rewrite_mean = row['mean_rewrite']
                    d = row['cohens_d']
                    p_value = row['p_value']
                    pct_change = row['pct_change']

                rows.append({
                    'condition': condition,
                    'model': model,
                    'marker': marker,
                    'category': info['category'],
                    'citation': info['citation'],
                    'orig_mean': orig_mean,
                    'rewrite_mean': rewrite_mean,
                    'cohens_d': d,
                    'p_value': p_value,
                    'pct_change': pct_change,
                })

    return pd.DataFrame(rows)


def main():
    print("Generating refined marker effects...")
    print(f"Computing {len(REFINED_MARKERS)} markers × {len(MODELS)} models × {len(CONDITIONS)} conditions")
    print()

    df = compute_effects()

    # Validate
    expected_rows = len(REFINED_MARKERS) * len(MODELS) * len(CONDITIONS)
    actual_rows = len(df)

    if actual_rows != expected_rows:
        print(f"Warning: Expected {expected_rows} rows, got {actual_rows}")
    else:
        print(f"✓ Generated {actual_rows} rows (13 markers × 3 models × 3 conditions)")

    # Summary statistics
    print()
    print("Summary by category (generic condition):")
    generic = df[df['condition'] == 'generic']
    for category in ['Function Words', 'Vocabulary', 'Syntax & Punctuation', 'Register']:
        cat_df = generic[generic['category'] == category]
        if len(cat_df) > 0:
            mean_d = cat_df['cohens_d'].abs().mean()
            n_markers = len(cat_df) // 3
            print(f"  {category}: {n_markers} markers, mean |d| = {mean_d:.2f}")

    # Direction analysis
    print()
    print("Direction agreement (all markers same direction across models):")
    for condition in CONDITIONS:
        cond_df = df[df['condition'] == condition]
        same_dir = 0
        for marker in REFINED_MARKERS:
            marker_df = cond_df[cond_df['marker'] == marker]
            if len(marker_df) == 3:
                signs = np.sign(marker_df['cohens_d'].values)
                if len(set(signs)) == 1:
                    same_dir += 1
        print(f"  {condition}: {same_dir}/{len(REFINED_MARKERS)} markers ({100*same_dir/len(REFINED_MARKERS):.0f}%)")

    # Save
    output_path = RESULTS_DIR / "refined_marker_effects.csv"
    df.to_csv(output_path, index=False)
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
