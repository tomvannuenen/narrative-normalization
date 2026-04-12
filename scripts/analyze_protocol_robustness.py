"""
Analyze protocol robustness test results.

Computes:
- Agreement rates between baseline and system protocols
- Correlation of marker values across protocols
- Assessment of protocol sensitivity

Usage:
    python scripts/analyze_protocol_robustness.py --prompt generic
    python scripts/analyze_protocol_robustness.py --prompt all

Author: Claude
Date: 2026-03-09
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REWRITE_MODELS, PROMPT_CONDITIONS

# Paths
DATA_DIR = Path("data")
PROTOCOL_DIR = DATA_DIR / "robustness_tests/protocol"
RESULTS_DIR = DATA_DIR / "robustness_tests/analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_protocol_robustness(prompt_condition: str):
    """
    Analyze protocol robustness for a single prompt condition.

    Args:
        prompt_condition: 'generic', 'voice_preserving', or 'rewrite_only'
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING PROTOCOL ROBUSTNESS: {prompt_condition.upper()}")
    print("=" * 80)

    # Load baseline and system protocol results
    baseline_file = PROTOCOL_DIR / f"protocol_{prompt_condition}_baseline.parquet"
    system_file = PROTOCOL_DIR / f"protocol_{prompt_condition}_system.parquet"

    if not baseline_file.exists():
        print(f"✗ Missing baseline file: {baseline_file}")
        return None

    if not system_file.exists():
        print(f"✗ Missing system file: {system_file}")
        return None

    baseline = pd.read_parquet(baseline_file)
    system = pd.read_parquet(system_file)

    print(f"✓ Loaded baseline protocol: {len(baseline)} stories")
    print(f"✓ Loaded system protocol: {len(system)} stories")

    # Analyze each model
    results_by_model = {}

    for model_info in REWRITE_MODELS:
        model_key = model_info['label']
        print(f"\n{'=' * 60}")
        print(f"Model: {model_key}")
        print(f"{'=' * 60}")

        baseline_col = f"rewrite_{model_key}_baseline"
        system_col = f"rewrite_{model_key}_system"

        if baseline_col not in baseline.columns or system_col not in system.columns:
            print(f"✗ Missing columns for {model_key}")
            continue

        # Compute markers for both protocols
        print(f"\nComputing markers for both protocols...")
        from src.markers import compute_all_markers

        # Baseline protocol markers
        print(f"  Baseline protocol...", end=" ")
        baseline_markers = []
        for idx, row in baseline.iterrows():
            text = row[baseline_col]
            if pd.notna(text) and str(text).strip():
                markers = compute_all_markers(text, include_empathic=False)
                markers['story_id'] = row['id']
                baseline_markers.append(markers)
        baseline_df = pd.DataFrame(baseline_markers)
        print(f"✓ {len(baseline_df)} stories")

        # System protocol markers
        print(f"  System protocol...", end=" ")
        system_markers = []
        for idx, row in system.iterrows():
            text = row[system_col]
            if pd.notna(text) and str(text).strip():
                markers = compute_all_markers(text, include_empathic=False)
                markers['story_id'] = row['id']
                system_markers.append(markers)
        system_df = pd.DataFrame(system_markers)
        print(f"✓ {len(system_df)} stories")

        # Merge on story_id
        merged = baseline_df.merge(system_df, on='story_id', suffixes=('_baseline', '_system'))
        print(f"✓ Aligned {len(merged)} stories across both protocols")

        # Compute agreement and correlation for each marker
        print(f"\nAnalyzing protocol stability...")

        marker_names = [col.replace('_baseline', '') for col in merged.columns
                       if col.endswith('_baseline') and col != 'story_id_baseline']

        stability_results = []

        for marker in marker_names:
            baseline_col_name = f"{marker}_baseline"
            system_col_name = f"{marker}_system"

            # Skip if columns don't exist
            if baseline_col_name not in merged.columns or system_col_name not in merged.columns:
                continue

            baseline_vals = merged[baseline_col_name].values
            system_vals = merged[system_col_name].values

            # Skip if all NaN
            if np.all(np.isnan(baseline_vals)) or np.all(np.isnan(system_vals)):
                continue

            # Compute correlation
            try:
                pearson_r, pearson_p = pearsonr(baseline_vals, system_vals)
            except:
                pearson_r, pearson_p = np.nan, np.nan

            try:
                spearman_r, spearman_p = spearmanr(baseline_vals, system_vals)
            except:
                spearman_r, spearman_p = np.nan, np.nan

            # Compute percent difference
            mean_baseline = np.nanmean(baseline_vals)
            mean_system = np.nanmean(system_vals)

            if mean_baseline != 0:
                pct_diff = ((mean_system - mean_baseline) / abs(mean_baseline)) * 100
            else:
                pct_diff = np.nan

            # Compute agreement (values within 10% of each other)
            if mean_baseline != 0:
                agreement = np.mean(np.abs((system_vals - baseline_vals) / baseline_vals) < 0.1) * 100
            else:
                agreement = np.nan

            # Assess stability
            if pearson_r >= 0.9:
                stability = "High"
            elif pearson_r >= 0.7:
                stability = "Moderate"
            else:
                stability = "Low"

            stability_results.append({
                'marker': marker,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'mean_baseline': mean_baseline,
                'mean_system': mean_system,
                'pct_diff': pct_diff,
                'agreement_10pct': agreement,
                'stability': stability,
            })

        results_df = pd.DataFrame(stability_results)
        results_df = results_df.sort_values('pearson_r', ascending=False)

        # Save results
        output_file = RESULTS_DIR / f"protocol_stability_{prompt_condition}_{model_key}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

        # Print summary
        print(f"\n{'─' * 60}")
        print(f"PROTOCOL STABILITY SUMMARY: {model_key}")
        print(f"{'─' * 60}")

        high = len(results_df[results_df['stability'] == 'High'])
        moderate = len(results_df[results_df['stability'] == 'Moderate'])
        low = len(results_df[results_df['stability'] == 'Low'])

        print(f"\nStability Distribution:")
        print(f"  High (r ≥ 0.9):      {high:3d} markers ({high/len(results_df)*100:.1f}%)")
        print(f"  Moderate (r 0.7-0.9): {moderate:3d} markers ({moderate/len(results_df)*100:.1f}%)")
        print(f"  Low (r < 0.7):       {low:3d} markers ({low/len(results_df)*100:.1f}%)")

        median_r = results_df['pearson_r'].median()
        median_diff = results_df['pct_diff'].abs().median()

        print(f"\nOverall Metrics:")
        print(f"  Median correlation (Pearson r): {median_r:.3f}")
        print(f"  Median absolute difference: {median_diff:.1f}%")

        # Most stable markers
        print(f"\nTop 10 Most Protocol-Stable Markers:")
        for idx, row in results_df.head(10).iterrows():
            print(f"  {row['marker']:40s}  r={row['pearson_r']:.3f}  Δ={row['pct_diff']:+5.1f}%")

        # Least stable markers
        print(f"\nBottom 10 Least Protocol-Stable Markers:")
        for idx, row in results_df.tail(10).iterrows():
            print(f"  {row['marker']:40s}  r={row['pearson_r']:.3f}  Δ={row['pct_diff']:+5.1f}%")

        results_by_model[model_key] = results_df

    # Overall decision
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    all_correlations = []
    for model_key, results_df in results_by_model.items():
        all_correlations.extend(results_df['pearson_r'].dropna().tolist())

    if all_correlations:
        median_r = np.median(all_correlations)
        pct_high = sum(1 for r in all_correlations if r >= 0.9) / len(all_correlations) * 100

        print(f"\nAcross all models and markers:")
        print(f"  Median correlation: {median_r:.3f}")
        print(f"  % High stability (r ≥ 0.9): {pct_high:.1f}%")

        print(f"\n{'─' * 80}")
        print("DECISION FRAMEWORK")
        print(f"{'─' * 80}")

        if median_r >= 0.9:
            print("\n✓ HIGH PROTOCOL STABILITY (Median r ≥ 0.9)")
            print("\nConclusion:")
            print("  - Effects are ROBUST across protocol designs")
            print("  - Findings generalize beyond specific API structure")
            print("  - Can confidently report results without protocol caveat")

        elif median_r >= 0.7:
            print("\n⚠ MODERATE PROTOCOL STABILITY (Median r 0.7-0.9)")
            print("\nConclusion:")
            print("  - Effects are GENERALLY stable but show some protocol sensitivity")
            print("  - Report protocol as boundary condition in paper")
            print("  - Note: 'Effects observed using USER-only message structure'")
            print("  - Consider testing additional protocol variants if critical")

        else:
            print("\n✗ LOW PROTOCOL STABILITY (Median r < 0.7)")
            print("\nConclusion:")
            print("  - Effects are HIGHLY protocol-dependent")
            print("  - Findings may be artifacts of specific API structure")
            print("  - MUST report as major limitation")
            print("  - Consider whether effects are genuine or methodological")

    return results_by_model


def main():
    parser = argparse.ArgumentParser(description="Analyze protocol robustness test results")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        choices=['generic', 'voice_preserving', 'rewrite_only', 'all'],
        help="Which prompt condition to analyze"
    )

    args = parser.parse_args()

    if args.prompt == 'all':
        prompts = ['generic', 'voice_preserving', 'rewrite_only']
    else:
        prompts = [args.prompt]

    for prompt in prompts:
        analyze_protocol_robustness(prompt)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
