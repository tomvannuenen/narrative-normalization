"""
Analyze temperature sensitivity test results.

Computes:
- Effect sizes (Cohen's d) for each marker at each temperature
- Pearson correlation of effect sizes across temperature conditions
- Assessment of whether normalization patterns are temperature-stable

High correlation (r > 0.85) indicates stable patterns regardless of temperature.
Low correlation suggests temperature-dependent artifacts.

Usage:
    python scripts/analyze_temperature_sensitivity.py

Author: Claude
Date: 2026-03-30
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REWRITE_MODELS, EXCLUDED_MARKERS

# Paths
DATA_DIR = Path("data")
TEMP_DIR = DATA_DIR / "robustness_tests/temperature"
RESULTS_DIR = DATA_DIR / "robustness_tests/analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.0, 0.7, 1.0]


def compute_cohens_d(original, rewrite):
    """
    Compute Cohen's d for paired samples.

    Args:
        original: array of original values
        rewrite: array of rewrite values

    Returns:
        float: Cohen's d effect size
    """
    diff = np.array(rewrite) - np.array(original)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0 or np.std(diff) == 0:
        return 0.0

    return np.mean(diff) / np.std(diff, ddof=1)


def analyze_temperature_sensitivity():
    """
    Analyze temperature sensitivity across all models.
    """
    print("\n" + "=" * 80)
    print("TEMPERATURE SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Load original stories for comparison
    sample = pd.read_parquet(DATA_DIR / "processed/sample.parquet")
    sample_ids = pd.read_csv(DATA_DIR / "robustness_tests/test1_self_consistency_sample.csv")
    originals = sample[sample['id'].isin(sample_ids['story_id'])].copy()

    print(f"\nLoaded {len(originals)} original stories")

    # Load temperature data
    temp_data = {}
    for temp in TEMPERATURES:
        temp_str = str(temp).replace('.', '_')
        file_path = TEMP_DIR / f"temperature_{temp_str}.parquet"
        if file_path.exists():
            temp_data[temp] = pd.read_parquet(file_path)
            print(f"✓ Loaded temperature {temp}: {len(temp_data[temp])} stories")
        else:
            print(f"✗ Missing temperature {temp} data: {file_path}")

    if len(temp_data) < 2:
        print("\n✗ Need at least 2 temperature conditions for comparison")
        return None

    # Compute original markers once
    print(f"\nComputing markers for original stories...")
    from src.markers import compute_all_markers

    orig_markers_list = []
    for idx, row in originals.iterrows():
        markers = compute_all_markers(row['story'], include_empathic=False)
        markers['story_id'] = row['id']
        orig_markers_list.append(markers)

    orig_markers = pd.DataFrame(orig_markers_list)
    print(f"✓ Computed {len(orig_markers.columns)-1} markers for {len(orig_markers)} stories")

    # Results for all models
    all_results = {}

    for model_info in REWRITE_MODELS:
        model_key = model_info['label']
        col_name = f"rewrite_{model_key}"

        print(f"\n{'=' * 60}")
        print(f"Model: {model_key}")
        print(f"{'=' * 60}")

        # Check data exists for all temperatures
        temps_available = [t for t in TEMPERATURES if t in temp_data and col_name in temp_data[t].columns]
        if len(temps_available) < 2:
            print(f"✗ Insufficient temperature data for {model_key}")
            continue

        print(f"Temperatures available: {temps_available}")

        # Compute markers for each temperature
        effect_sizes_by_temp = {}

        for temp in temps_available:
            print(f"\n  Computing markers for temperature {temp}...")
            df = temp_data[temp]

            markers_list = []
            for idx, row in df.iterrows():
                text = row[col_name]
                if pd.notna(text) and str(text).strip():
                    markers = compute_all_markers(text, include_empathic=False)
                    markers['story_id'] = row['id']
                    markers_list.append(markers)

            rewrite_markers = pd.DataFrame(markers_list)
            print(f"    ✓ Computed markers for {len(rewrite_markers)} stories")

            # Merge with originals
            merged = orig_markers.merge(rewrite_markers, on='story_id', suffixes=('_orig', '_rewrite'))

            # Compute effect sizes for each marker
            marker_names = [col.replace('_orig', '') for col in merged.columns
                          if col.endswith('_orig') and col != 'story_id_orig']

            effects = {}
            for marker in marker_names:
                if marker in EXCLUDED_MARKERS:
                    continue

                orig_col = f"{marker}_orig"
                rewrite_col = f"{marker}_rewrite"

                if orig_col in merged.columns and rewrite_col in merged.columns:
                    d = compute_cohens_d(merged[orig_col], merged[rewrite_col])
                    effects[marker] = d

            effect_sizes_by_temp[temp] = effects
            print(f"    ✓ Computed {len(effects)} effect sizes")

        # Compare effect sizes across temperatures
        print(f"\n  Comparing effect sizes across temperatures...")

        # Create comparison dataframe
        comparison_data = []
        markers_in_common = set.intersection(*[set(e.keys()) for e in effect_sizes_by_temp.values()])

        for marker in markers_in_common:
            row = {'marker': marker}
            for temp in temps_available:
                row[f'd_temp_{temp}'] = effect_sizes_by_temp[temp].get(marker, np.nan)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Compute pairwise correlations
        print(f"\n  Pairwise Correlations:")
        temp_pairs = [(temps_available[i], temps_available[j])
                     for i in range(len(temps_available))
                     for j in range(i+1, len(temps_available))]

        correlations = {}
        for t1, t2 in temp_pairs:
            col1, col2 = f'd_temp_{t1}', f'd_temp_{t2}'
            if col1 in comparison_df.columns and col2 in comparison_df.columns:
                valid = comparison_df[[col1, col2]].dropna()
                if len(valid) > 2:
                    r, p = stats.pearsonr(valid[col1], valid[col2])
                    correlations[(t1, t2)] = r
                    print(f"    r(temp={t1}, temp={t2}) = {r:.3f} (p={p:.4f})")

        # Save comparison results
        output_file = RESULTS_DIR / f"temperature_sensitivity_{model_key}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\n  ✓ Saved: {output_file}")

        # Summary statistics
        print(f"\n  {'─' * 50}")
        print(f"  SUMMARY: {model_key}")
        print(f"  {'─' * 50}")

        if correlations:
            mean_r = np.mean(list(correlations.values()))
            min_r = min(correlations.values())

            print(f"\n  Mean correlation across temp pairs: r = {mean_r:.3f}")
            print(f"  Minimum correlation: r = {min_r:.3f}")

            if min_r >= 0.85:
                print(f"\n  ✓ STABLE: Normalization patterns consistent across temperatures")
            elif min_r >= 0.70:
                print(f"\n  ⚠ MODERATE: Some temperature sensitivity detected")
            else:
                print(f"\n  ✗ UNSTABLE: Significant temperature dependence")

        # Show markers with largest temperature sensitivity
        if len(temps_available) >= 2:
            t1, t2 = temps_available[0], temps_available[-1]
            col1, col2 = f'd_temp_{t1}', f'd_temp_{t2}'
            comparison_df['diff'] = abs(comparison_df[col1] - comparison_df[col2])
            comparison_df = comparison_df.sort_values('diff', ascending=False)

            print(f"\n  Top 5 Most Temperature-Sensitive Markers:")
            for _, row in comparison_df.head(5).iterrows():
                print(f"    {row['marker']:40s}  "
                      f"d@{t1}={row[col1]:.2f}  d@{t2}={row[col2]:.2f}  Δ={row['diff']:.2f}")

            print(f"\n  Top 5 Most Temperature-Stable Markers:")
            for _, row in comparison_df.tail(5).iterrows():
                print(f"    {row['marker']:40s}  "
                      f"d@{t1}={row[col1]:.2f}  d@{t2}={row[col2]:.2f}  Δ={row['diff']:.2f}")

        all_results[model_key] = {
            'comparison_df': comparison_df,
            'correlations': correlations,
            'effect_sizes': effect_sizes_by_temp
        }

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    all_correlations = []
    for model_key, results in all_results.items():
        all_correlations.extend(results['correlations'].values())

    if all_correlations:
        overall_mean_r = np.mean(all_correlations)
        overall_min_r = min(all_correlations)

        print(f"\nAcross all models:")
        print(f"  Mean correlation: r = {overall_mean_r:.3f}")
        print(f"  Minimum correlation: r = {overall_min_r:.3f}")

        print(f"\n{'─' * 80}")
        print("DECISION")
        print(f"{'─' * 80}")

        if overall_min_r >= 0.85:
            print("\n✓ TEMPERATURE STABLE")
            print("\nConclusion:")
            print("  - Normalization patterns are consistent across temperature settings")
            print("  - Temperature=0.7 choice is well-justified")
            print("  - Results reflect model behavior, not temperature artifacts")

        elif overall_min_r >= 0.70:
            print("\n⚠ MODERATELY STABLE")
            print("\nConclusion:")
            print("  - Some temperature sensitivity detected")
            print("  - Core patterns (direction of effects) likely stable")
            print("  - Report temperature in methods; note as limitation")

        else:
            print("\n✗ TEMPERATURE DEPENDENT")
            print("\nConclusion:")
            print("  - Significant temperature dependence detected")
            print("  - Results may be temperature-specific")
            print("  - Consider running main study at multiple temperatures")
            print("  - Report temperature sensitivity prominently")

    # Save overall summary
    summary_file = RESULTS_DIR / "temperature_sensitivity_summary.csv"
    summary_rows = []
    for model_key, results in all_results.items():
        for (t1, t2), r in results['correlations'].items():
            summary_rows.append({
                'model': model_key,
                'temp_1': t1,
                'temp_2': t2,
                'correlation': r
            })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")

    return all_results


def main():
    results = analyze_temperature_sensitivity()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
