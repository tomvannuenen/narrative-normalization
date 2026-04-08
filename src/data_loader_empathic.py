"""Load and sample EmpathicStories dataset for the narrative normalization study."""

from __future__ import annotations

import pandas as pd

from src.config import (
    EMPATHIC_STORIES_RAW,
    PROCESSED_DIR,
    N_SAMPLES,
    MIN_WORD_COUNT,
    MAX_WORD_COUNT,
    RANDOM_SEED,
)


def load_empathic_stories() -> pd.DataFrame:
    """Load the EmpathicStories dataset from parquet.

    Returns:
        DataFrame with columns: id, story, word_count, Data Source, etc.
    """
    if not EMPATHIC_STORIES_RAW.exists():
        raise FileNotFoundError(
            f"EmpathicStories data not found at {EMPATHIC_STORIES_RAW}\n"
            f"Expected file: data/raw/empathic_stories_150plus.parquet"
        )

    df = pd.read_parquet(EMPATHIC_STORIES_RAW)
    print(f"Loaded {len(df)} stories from EmpathicStories dataset")
    print(f"Data sources: {df['Data Source'].value_counts().to_dict()}")

    return df


def filter_stories(df: pd.DataFrame) -> pd.DataFrame:
    """Filter stories by word count and clean text.

    Args:
        df: Raw EmpathicStories DataFrame

    Returns:
        Filtered DataFrame with stories in the valid word count range
    """
    # Filter by word count
    filtered = df[
        (df["word_count"] >= MIN_WORD_COUNT) &
        (df["word_count"] <= MAX_WORD_COUNT)
    ].copy()

    # Remove any stories with missing text
    filtered = filtered[filtered["story"].notna()].copy()
    filtered = filtered[filtered["story"].str.strip() != ""].copy()

    print(f"After filtering ({MIN_WORD_COUNT}-{MAX_WORD_COUNT} words): {len(filtered)} stories")

    return filtered


def sample_stories(df: pd.DataFrame, n: int = N_SAMPLES) -> pd.DataFrame:
    """Sample N stories from the dataset, stratified by data source.

    Args:
        df: Filtered EmpathicStories DataFrame
        n: Number of stories to sample (default from config)

    Returns:
        Sampled DataFrame with n stories
    """
    if len(df) <= n:
        print(f"Dataset has {len(df)} stories, returning all (requested {n})")
        return df.copy()

    # Stratified sampling by data source to maintain diversity
    sampled = df.groupby("Data Source", group_keys=False).apply(
        lambda x: x.sample(frac=n / len(df), random_state=RANDOM_SEED)
    )

    # If stratified sampling gives us fewer than n, top up with random samples
    if len(sampled) < n:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(n=n - len(sampled), random_state=RANDOM_SEED)
        sampled = pd.concat([sampled, extra])

    # If we got more than n (due to rounding), sample down
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=RANDOM_SEED)

    sampled = sampled.reset_index(drop=True)

    print(f"\nSampled {len(sampled)} stories")
    print(f"Source distribution:\n{sampled['Data Source'].value_counts()}")
    print(f"Word count stats:\n{sampled['word_count'].describe()}")

    return sampled


def save_sample(df: pd.DataFrame, tag: str = "sample") -> None:
    """Save sampled stories to processed directory.

    Args:
        df: Sampled DataFrame
        tag: Filename tag (default: 'sample')
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{tag}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved sample → {out_path}")


def load_sample(tag: str = "sample") -> pd.DataFrame:
    """Load previously saved sample.

    Args:
        tag: Filename tag (default: 'sample')

    Returns:
        Sampled DataFrame
    """
    path = PROCESSED_DIR / f"{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Sample not found at {path}\n"
            f"Run stage_sample() first to create the sample."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} stories from {path}")
    return df


def main():
    """Quick test/preview of the data loader."""
    print("=" * 60)
    print("EmpathicStories Data Loader")
    print("=" * 60)

    df = load_empathic_stories()
    filtered = filter_stories(df)
    sample = sample_stories(filtered, n=10)  # Small sample for testing

    print("\n" + "=" * 60)
    print("Sample Preview")
    print("=" * 60)
    for i, row in sample.head(3).iterrows():
        print(f"\n--- Story {i} ---")
        print(f"Source: {row['Data Source']}")
        print(f"Words: {row['word_count']}")
        print(f"Text: {row['story'][:200]}...")


if __name__ == "__main__":
    main()
