"""
Analyze self-consistency test results to measure baseline stochastic variation.

Computes:
- Intraclass Correlation Coefficient (ICC) for each marker
- Coefficient of Variation (CV) for each marker
- Reliability assessment (high/moderate/low)
- Decision on whether n=300 is sufficient

Usage:
    python scripts/analyze_self_consistency.py --prompt generic
    python scripts/analyze_self_consistency.py --prompt all

Author: Claude
Date: 2026-03-09
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REWRITE_MODELS, PROMPT_CONDITIONS

# Paths
DATA_DIR = Path("data")
CONSISTENCY_DIR = DATA_DIR / "robustness_tests/self_consistency"
RESULTS_DIR = DATA_DIR / "robustness_tests/analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_icc(data):
    """
    Compute Intraclass Correlation Coefficient (ICC) for consistency.

    ICC(3,1) = Between-subject variance / (Between-subject + Within-subject variance)

    High ICC (>0.9) = reliable, low stochastic variation
    Moderate ICC (0.7-0.9) = moderate reliability
    Low ICC (<0.7) = high stochastic variation

    Args:
        data: numpy array of shape (n_subjects, n_runs)

    Returns:
        float: ICC value between 0 and 1
    """
    n_subjects, n_runs = data.shape

    # Grand mean
    grand_mean = np.nanmean(data)

    # Between-subject sum of squares
    subject_means = np.nanmean(data, axis=1)
    ss_between = n_runs * np.nansum((subject_means - grand_mean) ** 2)

    # Within-subject sum of squares
    ss_within = np.nansum((data - subject_means[:, np.newaxis]) ** 2)

    # Mean squares
    df_between = n_subjects - 1
    df_within = n_subjects * (n_runs - 1)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # ICC
    icc = (ms_between - ms_within) / (ms_between + (n_runs - 1) * ms_within)

    return max(0, min(1, icc))  # Bound between 0 and 1


def compute_cv(data):
    """
    Compute Coefficient of Variation (CV) for each subject, then average.

    CV = (std / mean) * 100

    Low CV (<10%) = low variability
    Moderate CV (10-20%) = moderate variability
    High CV (>20%) = high variability

    Args:
        data: numpy array of shape (n_subjects, n_runs)

    Returns:
        float: Average CV across subjects
    """
    subject_means = np.nanmean(data, axis=1)
    subject_stds = np.nanstd(data, axis=1, ddof=1)

    # Avoid division by zero
    cvs = np.where(subject_means != 0,
                   (subject_stds / np.abs(subject_means)) * 100,
                   np.nan)

    return np.nanmean(cvs)


def analyze_consistency(prompt_condition: str):
    """
    Analyze self-consistency for a single prompt condition.

    Args:
        prompt_condition: 'generic', 'voice_preserving', or 'rewrite_only'
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING SELF-CONSISTENCY: {prompt_condition.upper()}")
    print("=" * 80)

    # Load the 3 runs
    runs = []
    for run_num in [1, 2, 3]:
        file_path = CONSISTENCY_DIR / f"consistency_{prompt_condition}_run{run_num}.parquet"
        if not file_path.exists():
            print(f"✗ Missing: {file_path}")
            return None
        runs.append(pd.read_parquet(file_path))
        print(f"✓ Loaded run {run_num}: {len(runs[-1])} stories")

    # For each model, compute markers and analyze consistency
    results_by_model = {}

    for model_info in REWRITE_MODELS:
        model_key = model_info['label']
        print(f"\n{'=' * 60}")
        print(f"Model: {model_key}")
        print(f"{'=' * 60}")

        # Get rewrite columns for all 3 runs
        run_cols = [f"rewrite_{model_key}_run{i}" for i in [1, 2, 3]]

        # Check columns exist in their respective run files
        cols_exist = all(run_cols[i] in runs[i].columns for i in range(3))
        if not cols_exist:
            print(f"✗ Missing rewrite columns for {model_key}")
            for i, col in enumerate(run_cols):
                status = "✓" if col in runs[i].columns else "✗"
                print(f"    {status} {col} in run{i+1}")
            continue

        # Compute markers for each run
        print(f"\nComputing markers for 3 runs...")
        from src.markers import compute_all_markers

        markers_by_run = []
        for run_idx, run_df in enumerate(runs, 1):
            col = f"rewrite_{model_key}_run{run_idx}"
            print(f"  Run {run_idx}...", end=" ")

            markers_list = []
            for idx, row in run_df.iterrows():
                text = row[col]
                if pd.notna(text) and str(text).strip():
                    markers = compute_all_markers(text, include_empathic=False)
                    markers['story_id'] = row['id']
                    markers_list.append(markers)

            markers_df = pd.DataFrame(markers_list)
            markers_by_run.append(markers_df)
            print(f"✓ {len(markers_df)} stories")

        # Align by story_id
        print(f"\nAligning markers across runs...")
        merged = markers_by_run[0].merge(
            markers_by_run[1], on='story_id', suffixes=('_r1', '_r2')
        ).merge(
            markers_by_run[2], on='story_id'
        )

        # Rename third run columns
        for col in markers_by_run[2].columns:
            if col != 'story_id' and col in merged.columns:
                merged.rename(columns={col: f"{col}_r3"}, inplace=True)

        print(f"✓ Aligned {len(merged)} stories across all 3 runs")

        # Compute ICC and CV for each marker
        print(f"\nComputing reliability metrics...")

        marker_names = [col.replace('_r1', '') for col in merged.columns if col.endswith('_r1')]

        reliability_results = []
        for marker in marker_names:
            cols = [f"{marker}_r1", f"{marker}_r2", f"{marker}_r3"]

            # Skip if columns don't exist
            if not all(c in merged.columns for c in cols):
                continue

            # Get data matrix (n_stories x 3_runs)
            data = merged[cols].values

            # Skip if all NaN
            if np.all(np.isnan(data)):
                continue

            # Compute ICC
            icc = compute_icc(data)

            # Compute CV
            cv = compute_cv(data)

            # Assess reliability
            if icc >= 0.9:
                reliability = "High"
            elif icc >= 0.7:
                reliability = "Moderate"
            else:
                reliability = "Low"

            # Compute means and ranges
            run_means = np.nanmean(data, axis=0)
            overall_mean = np.nanmean(run_means)
            overall_std = np.nanstd(data)

            reliability_results.append({
                'marker': marker,
                'icc': icc,
                'cv': cv,
                'reliability': reliability,
                'mean_r1': run_means[0],
                'mean_r2': run_means[1],
                'mean_r3': run_means[2],
                'overall_mean': overall_mean,
                'overall_std': overall_std,
            })

        results_df = pd.DataFrame(reliability_results)
        results_df = results_df.sort_values('icc', ascending=False)

        # Save results
        output_file = RESULTS_DIR / f"reliability_{prompt_condition}_{model_key}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

        # Print summary
        print(f"\n{'─' * 60}")
        print(f"RELIABILITY SUMMARY: {model_key}")
        print(f"{'─' * 60}")

        high = len(results_df[results_df['reliability'] == 'High'])
        moderate = len(results_df[results_df['reliability'] == 'Moderate'])
        low = len(results_df[results_df['reliability'] == 'Low'])

        print(f"\nReliability Distribution:")
        print(f"  High (ICC ≥ 0.9):      {high:3d} markers ({high/len(results_df)*100:.1f}%)")
        print(f"  Moderate (ICC 0.7-0.9): {moderate:3d} markers ({moderate/len(results_df)*100:.1f}%)")
        print(f"  Low (ICC < 0.7):       {low:3d} markers ({low/len(results_df)*100:.1f}%)")

        median_icc = results_df['icc'].median()
        median_cv = results_df['cv'].median()

        print(f"\nOverall Metrics:")
        print(f"  Median ICC: {median_icc:.3f}")
        print(f"  Median CV:  {median_cv:.1f}%")

        # Top 10 most reliable markers
        print(f"\nTop 10 Most Reliable Markers:")
        for idx, row in results_df.head(10).iterrows():
            print(f"  {row['marker']:40s}  ICC={row['icc']:.3f}  CV={row['cv']:5.1f}%")

        # Bottom 10 least reliable markers
        print(f"\nBottom 10 Least Reliable Markers:")
        for idx, row in results_df.tail(10).iterrows():
            print(f"  {row['marker']:40s}  ICC={row['icc']:.3f}  CV={row['cv']:5.1f}%")

        results_by_model[model_key] = results_df

    # Overall decision
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    all_iccs = []
    for model_key, results_df in results_by_model.items():
        all_iccs.extend(results_df['icc'].tolist())

    if all_iccs:
        median_icc = np.median(all_iccs)
        pct_high = sum(1 for icc in all_iccs if icc >= 0.9) / len(all_iccs) * 100

        print(f"\nAcross all models and markers:")
        print(f"  Median ICC: {median_icc:.3f}")
        print(f"  % High reliability (ICC ≥ 0.9): {pct_high:.1f}%")

        print(f"\n{'─' * 80}")
        print("DECISION FRAMEWORK")
        print(f"{'─' * 80}")

        if median_icc >= 0.9:
            print("\n✓ HIGH RELIABILITY (Median ICC ≥ 0.9)")
            print("\nRecommendation:")
            print("  - n=300 with 1 run per condition is SUFFICIENT")
            print("  - Prompt effects exceed baseline stochastic variation")
            print("  - Proceed with main analysis confidently")

        elif median_icc >= 0.7:
            print("\n⚠ MODERATE RELIABILITY (Median ICC 0.7-0.9)")
            print("\nRecommendation:")
            print("  - n=300 is acceptable but interpretation should be cautious")
            print("  - Consider expanding to n=600 OR adding 2-3 runs for subset")
            print("  - Report ICC as benchmark in paper")

        else:
            print("\n✗ LOW RELIABILITY (Median ICC < 0.7)")
            print("\nRecommendation:")
            print("  - n=300 with 1 run is INSUFFICIENT")
            print("  - MUST add multiple runs (3+ runs) for main study")
            print("  - OR significantly expand sample size to n=1000+")
            print("  - Current design cannot distinguish effects from noise")

    return results_by_model


def main():
    parser = argparse.ArgumentParser(description="Analyze self-consistency test results")
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
        analyze_consistency(prompt)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
