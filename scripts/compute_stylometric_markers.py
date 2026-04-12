#!/usr/bin/env python3
"""Compute core stylometric markers for originals and rewrites.

This script computes the new stylometric markers (character n-grams,
vocabulary richness, word-length distribution, Delta features) following
Stamatatos (2009), Eder et al. (2016), and related stylometric literature.

Usage:
    python scripts/compute_stylometric_markers.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.stylometric_markers import compute_stylometric_markers


def compute_markers_for_texts(texts: list[str], desc: str = "Computing markers") -> pd.DataFrame:
    """Compute stylometric markers for a list of texts."""
    results = []
    for text in tqdm(texts, desc=desc):
        if pd.isna(text) or not text.strip():
            results.append({})
        else:
            results.append(compute_stylometric_markers(text))
    return pd.DataFrame(results)


def main():
    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "processed" / "stylometric"
    output_dir.mkdir(exist_ok=True)

    # Process each condition
    conditions = ["generic", "voice_preserving", "rewrite_only"]
    models = ["gpt54", "claude_sonnet", "gemini_31_pro"]

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Processing condition: {condition}")
        print(f"{'='*60}")

        # Load rewrites data
        rewrite_file = data_dir / f"rewrites_{condition}.parquet"
        if not rewrite_file.exists():
            print(f"  Skipping {condition} - file not found")
            continue

        df = pd.read_parquet(rewrite_file)
        print(f"  Loaded {len(df)} stories")

        # Compute markers for original stories (only once per condition)
        print("\n  Computing markers for ORIGINAL stories...")
        original_markers = compute_markers_for_texts(
            df['story'].tolist(),
            desc="  Original stories"
        )
        original_markers['id'] = df['id'].values
        original_markers['type'] = 'original'
        original_markers.to_parquet(
            output_dir / f"stylometric_original_{condition}.parquet"
        )
        print(f"  Saved original markers: {len(original_markers)} rows, {len(original_markers.columns)} features")

        # Compute markers for each model's rewrites
        for model in models:
            col_name = f"rewrite_{model}"
            if col_name not in df.columns:
                print(f"  Skipping {model} - column not found")
                continue

            print(f"\n  Computing markers for {model.upper()} rewrites...")
            rewrite_markers = compute_markers_for_texts(
                df[col_name].tolist(),
                desc=f"  {model}"
            )
            rewrite_markers['id'] = df['id'].values
            rewrite_markers['type'] = f'rewrite_{model}'
            rewrite_markers.to_parquet(
                output_dir / f"stylometric_{condition}_{model}.parquet"
            )
            print(f"  Saved {model} markers: {len(rewrite_markers)} rows, {len(rewrite_markers.columns)} features")

    print("\n" + "="*60)
    print("SUMMARY: Stylometric feature categories computed:")
    print("="*60)
    print("""
    1. Character n-grams (2,3,4-grams): entropy, hapax ratio, unique ratio
       - Following Stamatatos (2009, 2013)

    2. MFW profile: coverage by top-10/50 words, concentration (Gini), FW ratio
       - Following Burrows (2002), Eder et al. (2016)

    3. Vocabulary richness: Yule's K, Simpson's D, Honore's R, Sichel's S, Brunet's W
       - Following Tweedie & Baayen (1998)

    4. Word-length distribution: proportions for lengths 1-15, mean, std, skew
       - Following Mendenhall (1887)

    5. Punctuation patterns: comma, semicolon, colon, dash, etc. ratios
       - Standard stylometric features

    6. Sentence-length distribution: mean, std, median, short/long ratios
       - Standard stylometric features

    7. Delta features: FW z-score statistics, deviation from expected
       - Following Burrows (2002)
    """)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
