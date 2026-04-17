#!/usr/bin/env python3
"""Compute experiential→explanatory markers for narrative stance analysis.

This script computes the marker set designed to operationalize the shift
from experiential narration to explanatory narration when LLMs rewrite
personal narratives.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stance_markers import compute_indexical_markers


def compute_markers_for_texts(texts: list[str], desc: str = "Computing markers") -> pd.DataFrame:
    """Compute all indexical markers for a list of texts.

    This calls compute_indexical_markers which computes markers from all 6 dimensions:
    epistemic_stance, narratorial_involvement, orality_literariness,
    temporal_causal_structure, affective_positioning, experiential_explanatory
    """
    results = []
    for text in tqdm(texts, desc=desc):
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            # Return NaN for empty/invalid texts
            results.append({})
        else:
            try:
                markers = compute_indexical_markers(text)
                results.append(markers)
            except Exception as e:
                print(f"  Warning: Failed to compute markers: {e}")
                results.append({})
    return pd.DataFrame(results)


def cohens_d(orig: np.ndarray, rewrite: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diff = orig - rewrite
    pooled_std = np.sqrt((orig.std()**2 + rewrite.std()**2) / 2)
    if pooled_std == 0:
        return 0.0
    return diff.mean() / pooled_std


def main():
    data_dir = Path("data/processed")
    output_dir = Path("results/stance_markers")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sample (original texts)
    print("Loading data...")
    sample = pd.read_parquet(data_dir / "sample.parquet")
    rewrites_generic = pd.read_parquet(data_dir / "rewrites_generic.parquet")
    rewrites_rewrite_only = pd.read_parquet(data_dir / "rewrites_rewrite_only.parquet")
    rewrites_voice = pd.read_parquet(data_dir / "rewrites_voice_preserving.parquet")

    # Compute markers for originals
    print("\n=== Computing markers for original texts ===")
    orig_markers = compute_markers_for_texts(sample["story"].tolist(), "Original texts")
    orig_markers.to_parquet(output_dir / "markers_original.parquet")
    print(f"Saved {len(orig_markers)} original marker rows")

    # Models and conditions
    models = ["gpt54", "claude_sonnet", "gemini_31_pro"]
    conditions = [
        ("generic", rewrites_generic),
        ("rewrite_only", rewrites_rewrite_only),
        ("voice_preserving", rewrites_voice),
    ]

    results = []

    for condition_name, rewrites_df in conditions:
        print(f"\n=== Processing {condition_name} condition ===")

        for model in models:
            col = f"rewrite_{model}"
            if col not in rewrites_df.columns:
                print(f"  Skipping {model} - column not found")
                continue

            print(f"\n  Computing markers for {model}...")
            rewrite_markers = compute_markers_for_texts(
                rewrites_df[col].tolist(),
                f"{model} ({condition_name})"
            )
            rewrite_markers.to_parquet(output_dir / f"markers_{condition_name}_{model}.parquet")

            # Calculate effect sizes
            print(f"  Calculating effect sizes for {model}...")
            marker_cols = [c for c in orig_markers.columns if c.startswith("idx_")]

            for marker in marker_cols:
                if marker not in rewrite_markers.columns:
                    continue

                # Get valid pairs (both original and rewrite have values)
                valid_mask = ~(orig_markers[marker].isna() | rewrite_markers[marker].isna())
                orig_vals = orig_markers.loc[valid_mask, marker].values
                rewrite_vals = rewrite_markers.loc[valid_mask, marker].values

                if len(orig_vals) < 10:
                    continue

                d = cohens_d(orig_vals, rewrite_vals)

                results.append({
                    "condition": condition_name,
                    "model": model,
                    "marker": marker,
                    "d": d,
                    "orig_mean": orig_vals.mean(),
                    "rewrite_mean": rewrite_vals.mean(),
                    "n": len(orig_vals),
                })

    # Save effect size results
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_dir / "effect_sizes.parquet")
    results_df.to_csv(output_dir / "effect_sizes.csv", index=False)
    print(f"\n=== Saved {len(results_df)} effect size calculations ===")

    # Print summary table
    print("\n=== SUMMARY: Mean Effect Sizes Across Models (Generic Condition) ===\n")
    generic = results_df[results_df["condition"] == "generic"]
    pivot = generic.pivot_table(index="marker", columns="model", values="d", aggfunc="mean")
    pivot["mean_d"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_d")
    print(pivot.to_string())

    # Print comparison across conditions
    print("\n=== SUMMARY: Effect Sizes by Condition (Averaged Across Models) ===\n")
    by_condition = results_df.groupby(["condition", "marker"])["d"].mean().reset_index()
    pivot_cond = by_condition.pivot_table(index="marker", columns="condition", values="d")
    if "generic" in pivot_cond.columns:
        pivot_cond = pivot_cond.sort_values("generic")
    print(pivot_cond.to_string())


if __name__ == "__main__":
    main()
