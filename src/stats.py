"""Statistical comparison of original vs. rewritten texts.

All tests are paired (each story is compared against its own rewrite),
so we use Wilcoxon signed-rank tests and compute effect sizes (Cohen's d
on the paired differences).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import ALPHA, EXCLUDED_MARKERS


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = a - b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    return float(d)


def compare_markers(
    df_orig: pd.DataFrame,
    df_rewrite: pd.DataFrame,
    marker_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run paired Wilcoxon tests + effect sizes for every shared numeric column.

    Parameters
    ----------
    df_orig : marker values for original texts (one row per story).
    df_rewrite : marker values for rewritten texts (same order).
    marker_cols : columns to compare. If None, all shared numeric columns.

    Returns
    -------
    DataFrame with columns: marker, mean_orig, mean_rewrite, mean_diff,
    wilcoxon_stat, p_value, significant, cohens_d, direction.
    """
    if marker_cols is None:
        shared = set(df_orig.columns) & set(df_rewrite.columns)
        marker_cols = sorted(
            c for c in shared
            if pd.api.types.is_numeric_dtype(df_orig[c])
            and c not in EXCLUDED_MARKERS
        )

    rows = []
    for col in marker_cols:
        a = df_orig[col].to_numpy(dtype=float)
        b = df_rewrite[col].to_numpy(dtype=float)

        # Drop pairs where either is NaN
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]

        if len(a) < 10:
            continue

        mean_orig = float(a.mean())
        mean_rewrite = float(b.mean())
        mean_diff = float((a - b).mean())

        try:
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, p = 0.0, 1.0

        d = cohens_d_paired(a, b)

        direction = "decrease" if mean_diff > 0 else "increase" if mean_diff < 0 else "none"

        rows.append({
            "marker": col,
            "mean_orig": round(mean_orig, 4),
            "mean_rewrite": round(mean_rewrite, 4),
            "mean_diff": round(mean_diff, 4),
            "pct_change": round((mean_rewrite - mean_orig) / abs(mean_orig) * 100, 2) if mean_orig != 0 else 0.0,
            "wilcoxon_stat": round(float(stat), 2),
            "p_value": float(p),
            "significant": p < ALPHA,
            "cohens_d": round(d, 4),
            "direction": direction,
        })

    result = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction
    if len(result) > 0:
        from statsmodels.stats.multitest import multipletests
        reject, pvals_corrected, _, _ = multipletests(
            result["p_value"], alpha=ALPHA, method="fdr_bh"
        )
        result["p_value_fdr"] = pvals_corrected
        result["significant_fdr"] = reject

    return result.sort_values("p_value").reset_index(drop=True)


def summary_by_dimension(comparison: pd.DataFrame) -> pd.DataFrame:
    """Group markers by dimension prefix and summarise effects."""
    def _dim(marker: str) -> str:
        prefix = marker.split("_")[0]
        return {
            "ld": "lexical_diversity",
            "syn": "syntactic_complexity",
            "sd": "semantic_distance",
            "ent": "textual_entropy",
            "sent": "sentiment_affect",
            "coh": "discourse_cohesion",
            "sty": "stylometric",
            "pos": "stylometric",
            "read": "readability",
        }.get(prefix, "other")

    df = comparison.copy()
    df["dimension"] = df["marker"].apply(_dim)

    summary = df.groupby("dimension").agg(
        n_markers=("marker", "count"),
        n_significant=("significant_fdr", "sum"),
        mean_abs_cohens_d=("cohens_d", lambda x: round(abs(x).mean(), 4)),
        markers_decreased=("direction", lambda x: (x == "decrease").sum()),
        markers_increased=("direction", lambda x: (x == "increase").sum()),
    ).reset_index()

    return summary.sort_values("mean_abs_cohens_d", ascending=False)
