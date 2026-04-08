#!/usr/bin/env python3
"""Main pipeline for the EmpathicStories narrative normalization study.

This study investigates how LLMs operationalize the instruction to "improve"
personal narratives, testing whether they preserve empathic structure or
normalize stories toward model-preferred templates.

Usage
-----
    # Run full pipeline with all 3 prompt conditions (300 stories × 3 models × 3 prompts):
    python run_empathic_pipeline.py

    # Run with specific prompt condition:
    python run_empathic_pipeline.py --prompt generic
    python run_empathic_pipeline.py --prompt voice
    python run_empathic_pipeline.py --prompt rewrite

    # Run individual stages:
    python run_empathic_pipeline.py --stage sample
    python run_empathic_pipeline.py --stage rewrite
    python run_empathic_pipeline.py --stage analyze
    python run_empathic_pipeline.py --stage compare

    # Test run (small sample):
    python run_empathic_pipeline.py --test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    REWRITE_MODELS,
    PROMPT_CONDITIONS,
    N_SAMPLES,
)


# ── Stage 1: Sample stories ───────────────────────────────────────────

def stage_sample(n_samples: int = N_SAMPLES):
    from src.data_loader_empathic import (
        load_empathic_stories,
        filter_stories,
        sample_stories,
        save_sample,
    )

    print("=" * 60)
    print("STAGE 1: Sampling stories from EmpathicStories dataset")
    print("=" * 60)

    df = load_empathic_stories()
    filtered = filter_stories(df)
    sample = sample_stories(filtered, n=n_samples)
    save_sample(sample)

    print(f"\nSampled {len(sample)} stories.")
    return sample


# ── Stage 2: Rewrite with LLMs (multi-prompt) ────────────────────────

def stage_rewrite(prompt_condition: str | None = None, mode: str = "batch", n_test: int | None = None):
    """Rewrite stories with all models and prompt conditions.

    Args:
        prompt_condition: Specific prompt to use ('generic', 'voice', 'rewrite'),
                         or None to run all conditions
        mode: 'batch' (50% cheaper) or 'sync' (one-at-a-time)
        n_test: If set, only rewrite this many stories (sync mode, no checkpoint saved)
    """
    from src.data_loader_empathic import load_sample
    from src.rewriter import rewrite_stories, save_rewrites
    from src.config import PROMPT_CONDITIONS

    print("=" * 60)
    print(f"STAGE 2: Rewriting stories with LLMs (mode={mode})")
    print("=" * 60)

    sample = load_sample("sample")

    if n_test:
        sample = sample.head(n_test)
        mode = "sync"
        print(f"TEST MODE: using {n_test} stories, sync mode (no checkpoint written)")

    # Determine which prompt conditions to run
    if prompt_condition:
        if prompt_condition not in PROMPT_CONDITIONS:
            raise ValueError(
                f"Invalid prompt condition: {prompt_condition}\n"
                f"Valid options: {list(PROMPT_CONDITIONS.keys())}"
            )
        conditions = {prompt_condition: PROMPT_CONDITIONS[prompt_condition]}
    else:
        conditions = PROMPT_CONDITIONS

    print(f"\nRunning {len(conditions)} prompt condition(s):")
    for cond_name, cond in conditions.items():
        print(f"  - {cond_name}: {cond['description']}")

    # Run rewriting for each prompt condition
    for cond_name, cond in conditions.items():
        print(f"\n{'=' * 60}")
        print(f"Prompt Condition: {cond_name}")
        print(f"Prompt: {cond['prompt'][:100]}...")
        print(f"{'=' * 60}")

        # Pass prompt directly to rewrite_stories (fixes import bug)
        ckpt_tag = f"rewrite_checkpoint_{cond_name}" if not n_test else f"rewrite_checkpoint_{cond_name}_test"
        result = rewrite_stories(sample, mode=mode, prompt=cond["prompt"], ckpt_tag=ckpt_tag)

        # Save with condition-specific tag (test runs go to a separate file)
        tag = f"rewrites_{cond_name}" if not n_test else f"rewrites_{cond_name}_test"
        save_rewrites(result, tag=tag)

    print("\nAll rewrite conditions completed.")


# ── Stage 3: Compute linguistic markers ────────────────────────────────

def stage_analyze(prompt_condition: str | None = None, checkpoint_every: int = 25):
    """Compute markers for all text versions (original + rewrites).

    Args:
        prompt_condition: Analyze specific condition, or None for all
        checkpoint_every: Save checkpoint every N rows
    """
    from src.markers import compute_all_markers

    print("=" * 60)
    print("STAGE 3: Computing linguistic markers")
    print("=" * 60)

    # Determine which conditions to analyze
    if prompt_condition:
        conditions = [prompt_condition]
    else:
        conditions = list(PROMPT_CONDITIONS.keys())

    # Analyze original stories (shared across all conditions)
    _analyze_text_column("sample", "story", checkpoint_every, include_empathic=True)

    # Analyze each rewrite condition
    for cond_name in conditions:
        tag = f"rewrites_{cond_name}"
        rewrite_path = PROCESSED_DIR / f"{tag}.parquet"

        if not rewrite_path.exists():
            print(f"\n[{cond_name}] Rewrites not found at {rewrite_path}. Skipping.")
            continue

        df = pd.read_parquet(rewrite_path)

        # Analyze each model's rewrites for this condition
        for m in REWRITE_MODELS:
            col = f"rewrite_{m['label']}"
            if col in df.columns:
                output_tag = f"{cond_name}_{m['label']}"
                _analyze_text_column(
                    tag,
                    col,
                    checkpoint_every,
                    include_empathic=True,
                    output_tag=output_tag,
                )
                # Add semantic voice distance (requires paired original + rewrite)
                _compute_semantic_distances(
                    sample_path=PROCESSED_DIR / "sample.parquet",
                    rewrites_df=df,
                    rewrite_col=col,
                    markers_tag=output_tag,
                )

    print("\nMarker computation complete.")


def _analyze_text_column(
    data_tag: str,
    column: str,
    checkpoint_every: int,
    include_empathic: bool = True,
    output_tag: str | None = None,
):
    """Helper to compute markers for a single text column."""
    from src.markers import compute_all_markers

    data_path = PROCESSED_DIR / f"{data_tag}.parquet"
    df = pd.read_parquet(data_path)

    # Output tag defaults to column name
    if output_tag is None:
        output_tag = column

    out_path = PROCESSED_DIR / f"markers_{output_tag}.parquet"

    # Resume support
    existing_ids = set()
    records = []
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        if len(existing_df) == len(df):
            print(f"[{output_tag}] markers already computed. Skipping.")
            return
        existing_ids = set(existing_df["id"].tolist())
        records = existing_df.to_dict("records")
        print(f"[{output_tag}] Resuming from checkpoint ({len(existing_ids)}/{len(df)} done).")

    remaining = [(i, row) for i, row in df.iterrows() if row["id"] not in existing_ids]
    if not remaining:
        print(f"[{output_tag}] All markers computed.")
        return

    print(f"\n[{output_tag}] Computing markers for {len(remaining)} texts…")
    for count, (i, row) in enumerate(tqdm(remaining, desc=output_tag), 1):
        text = row[column]
        if pd.isna(text) or not str(text).strip():
            records.append({"id": row["id"]})
        else:
            markers = compute_all_markers(text, include_empathic=include_empathic)
            markers["id"] = row["id"]
            records.append(markers)

        if count % checkpoint_every == 0:
            markers_df = pd.DataFrame(records)
            markers_df.to_parquet(out_path, index=False)
            tqdm.write(f"  [{output_tag}] checkpoint at {len(records)}/{len(df)}")

    markers_df = pd.DataFrame(records)
    markers_df.to_parquet(out_path, index=False)
    print(f"[{output_tag}] Saved → {out_path}")


def _compute_semantic_distances(
    sample_path,
    rewrites_df: pd.DataFrame,
    rewrite_col: str,
    markers_tag: str,
):
    """Compute semantic voice distance and merge into an existing markers file.

    Loads original stories and rewrites, computes pairwise cosine similarity
    using sentence-transformers, and adds 'semantic_voice_distance' to the
    markers parquet for this rewrite condition.
    """
    from src.empathic_markers import compute_semantic_distances

    markers_path = PROCESSED_DIR / f"markers_{markers_tag}.parquet"
    if not markers_path.exists():
        print(f"  [{markers_tag}] Markers file not found, skipping semantic distance.")
        return

    markers_df = pd.read_parquet(markers_path)

    if "semantic_voice_distance" in markers_df.columns:
        print(f"  [{markers_tag}] Semantic distances already computed. Skipping.")
        return

    sample_df = pd.read_parquet(sample_path)

    # Align on id
    merged = markers_df[["id"]].merge(
        sample_df[["id", "story"]].merge(
            rewrites_df[["id", rewrite_col]], on="id"
        ),
        on="id",
    )

    valid = merged[merged[rewrite_col].notna() & merged["story"].notna()]
    print(f"  [{markers_tag}] Computing semantic distances for {len(valid)} pairs…")

    similarities = compute_semantic_distances(
        valid["story"].tolist(),
        valid[rewrite_col].tolist(),
    )

    sim_series = pd.Series(similarities, index=valid.index)
    markers_df["semantic_voice_distance"] = sim_series
    markers_df.to_parquet(markers_path, index=False)
    print(f"  [{markers_tag}] Semantic distances saved → {markers_path}")


# ── Stage 4: Statistical comparison ────────────────────────────────────

def stage_compare(prompt_condition: str | None = None):
    """Run statistical comparisons and generate visualizations.

    Args:
        prompt_condition: Compare specific condition, or None for all
    """
    from src.stats import compare_markers, summary_by_dimension
    from src.visualize import (
        plot_effect_sizes,
        plot_dimension_summary,
        plot_paired_distributions,
    )

    print("=" * 60)
    print("STAGE 4: Statistical comparison & visualization")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load original markers (shared across conditions)
    orig_path = PROCESSED_DIR / "markers_story.parquet"
    if not orig_path.exists():
        print("Original markers not found. Run --stage analyze first.")
        return

    orig = pd.read_parquet(orig_path)

    # Determine which conditions to compare
    if prompt_condition:
        conditions = [prompt_condition]
    else:
        conditions = list(PROMPT_CONDITIONS.keys())

    for cond_name in conditions:
        print(f"\n{'=' * 60}")
        print(f"Analyzing Prompt Condition: {cond_name}")
        print(f"{'=' * 60}")

        for m in REWRITE_MODELS:
            label = m["label"]
            rewrite_tag = f"{cond_name}_{label}"
            rewrite_path = PROCESSED_DIR / f"markers_{rewrite_tag}.parquet"

            if not rewrite_path.exists():
                print(f"[{rewrite_tag}] Markers not found. Skipping.")
                continue

            rewrite = pd.read_parquet(rewrite_path)

            print(f"\n{'-' * 40}")
            print(f"Comparing: original vs. {rewrite_tag}")
            print(f"{'-' * 40}")

            comparison = compare_markers(orig, rewrite)
            comparison.to_csv(
                RESULTS_DIR / f"comparison_{cond_name}_{label}.csv",
                index=False,
            )

            summary = summary_by_dimension(comparison)
            summary.to_csv(
                RESULTS_DIR / f"summary_{cond_name}_{label}.csv",
                index=False,
            )

            # Print highlights
            sig = comparison[comparison.get("significant_fdr", comparison["significant"])]
            print(f"\n{len(sig)} significant markers (of {len(comparison)}):\n")
            for _, row in sig.head(10).iterrows():
                arrow = "↓" if row["direction"] == "decrease" else "↑"
                print(
                    f"  {arrow} {row['marker']:40s}  "
                    f"d={row['cohens_d']:+.3f}  "
                    f"Δ={row['pct_change']:+.1f}%  "
                    f"p={row['p_value']:.2e}"
                )

            # Plots
            plot_effect_sizes(comparison, f"{cond_name}_{label}")
            plot_dimension_summary(summary, f"{cond_name}_{label}")

    print("\nDone.")


# ── CLI ────────────────────────────────────────────────────────────────

STAGES = {
    "sample": stage_sample,
    "rewrite": stage_rewrite,
    "analyze": stage_analyze,
    "compare": stage_compare,
}


def main():
    parser = argparse.ArgumentParser(
        description="EmpathicStories Narrative Normalization Study Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a single stage. Omit to run the full pipeline.",
    )
    parser.add_argument(
        "--prompt",
        choices=list(PROMPT_CONDITIONS.keys()),
        default=None,
        help="Run specific prompt condition only (generic/voice/rewrite). "
             "Omit to run all conditions.",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "sync"],
        default="batch",
        help="Rewrite mode: 'batch' (50%% cheaper, default) or 'sync'.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode with only 10 stories",
    )

    args = parser.parse_args()

    # Test mode: use smaller sample
    n_samples = 10 if args.test else N_SAMPLES
    if args.test:
        print(f"\n{'*' * 60}")
        print(f"TEST MODE: Using 3 stories per model, sync mode")
        print(f"{'*' * 60}\n")

    if args.stage:
        if args.stage == "sample":
            stage_sample(n_samples=n_samples)
        elif args.stage == "rewrite":
            stage_rewrite(prompt_condition=args.prompt, mode=args.mode,
                          n_test=3 if args.test else None)
        elif args.stage == "analyze":
            stage_analyze(prompt_condition=args.prompt)
        elif args.stage == "compare":
            stage_compare(prompt_condition=args.prompt)
    else:
        # Full pipeline
        stage_sample(n_samples=n_samples)
        stage_rewrite(prompt_condition=args.prompt, mode=args.mode)
        stage_analyze(prompt_condition=args.prompt)
        stage_compare(prompt_condition=args.prompt)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
