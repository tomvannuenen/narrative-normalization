#!/usr/bin/env python3
"""Generate tables from the DSH paper using computed results.

Creates all tables from results.tex:
- Table 1: Marker-level change under generic improvement prompt
- Table 2: Comparison across prompt conditions
- Table 3: Dimension-level effects
- Table 4: Effect sizes across prompt conditions
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths (relative to repo root, run from repo root directory)
RESULTS_DIR = Path("results")

MODELS = ['gpt54', 'claude_sonnet', 'gemini_31_pro']
CONDITIONS = ['generic', 'rewrite_only', 'voice_preserving']

MODEL_LABELS = {
    'gpt54': 'GPT-5.4',
    'claude_sonnet': 'Claude Sonnet 4.6',
    'gemini_31_pro': 'Gemini 3.1 Pro',
}

# All 10 dimensions ordered by theoretical relevance for DH
DIMENSION_MARKERS = {
    'Voice Markers': ['voice_contraction_density', 'voice_first_person_density', 'voice_disfluency', 'voice_proper_noun_density', 'voice_specific_number_density'],
    'Lexical Diversity': ['ld_unique_words', 'ld_mtld', 'ld_hapax_ratio'],
    'Stylometric': ['pos_adj_ratio', 'pos_adv_ratio', 'pos_noun_ratio', 'pos_pron_ratio', 'pos_punct_ratio', 'sty_exclamation_per_sent', 'sty_function_word_ratio', 'sty_question_per_sent'],
    'Readability': ['read_flesch_ease', 'read_coleman_liau', 'read_syllables_per_word', 'sty_mean_word_length'],
    'Syntactic Complexity': ['syn_mean_sent_len', 'syn_std_sent_len', 'syn_n_sents', 'syn_subordination_ratio', 'syn_mean_depth', 'syn_min_sent_len'],
    'Semantic Coherence': ['coh_local_similarity', 'coh_global_similarity', 'sd_max_consecutive', 'sd_spread', 'sd_std_consecutive'],
    'Emotional Dynamics': ['emo_granularity', 'emo_volatility', 'emo_word_density'],
    'Sentiment': ['sent_mean_compound', 'sent_std_compound', 'sent_pct_positive', 'sent_pct_negative', 'sent_min_compound', 'sent_max_compound'],
    'Narrative Structure': ['narr_past_tense_ratio', 'narr_action_verb_density', 'narr_temporal_marker_density', 'causal_markers', 'moral_loyalty', 'conflict_second_person'],
    'Textual Entropy': ['ent_char', 'ent_word'],
}


def load_comparison(condition: str, model: str) -> pd.DataFrame:
    """Load a comparison CSV."""
    path = RESULTS_DIR / f"comparison_{condition}_{model}.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Exclude 'id' if present (not a marker)
        df = df[df['marker'] != 'id']
        return df
    return None


def generate_table1():
    """Table 1: Marker-level change under generic improvement prompt.

    Shows: Model, Sig. Markers (count/total), Mean |d|
    """
    print("\n" + "="*60)
    print("TABLE 1: Marker-level change under generic improvement prompt")
    print("="*60)
    print(f"{'Model':<25} {'Sig. Markers':<15} {'Mean |d|':<10}")
    print("-"*50)

    for model in MODELS:
        df = load_comparison('generic', model)
        if df is not None:
            n_sig = df['significant_fdr'].sum()
            n_total = len(df)
            pct = 100 * n_sig / n_total
            mean_d = df['cohens_d'].abs().mean()
            print(f"{MODEL_LABELS[model]:<25} {n_sig}/{n_total} ({pct:.0f}%){'':5} {mean_d:.2f}")

    print("\nNote: Mean |d| denotes the mean absolute Cohen's d across retained markers.")


def generate_table2():
    """Table 2: Comparison across prompt conditions.

    Shows: Condition, Mean Sig. %, Mean |d|, Direction Agreement
    """
    print("\n" + "="*60)
    print("TABLE 2: Comparison across prompt conditions")
    print("="*60)
    print(f"{'Condition':<30} {'Mean Sig. %':<15} {'Mean |d|':<12} {'Direction Agreement':<20}")
    print("-"*75)

    # Get generic condition data for direction comparison
    generic_directions = {}
    for model in MODELS:
        df = load_comparison('generic', model)
        if df is not None:
            for _, row in df.iterrows():
                marker = row['marker']
                if marker not in generic_directions:
                    generic_directions[marker] = []
                generic_directions[marker].append(row['cohens_d'] > 0)  # True = decrease

    for cond in CONDITIONS:
        sig_pcts = []
        mean_ds = []
        direction_agreements = []

        for model in MODELS:
            df = load_comparison(cond, model)
            if df is not None:
                sig_pcts.append(100 * df['significant_fdr'].sum() / len(df))
                mean_ds.append(df['cohens_d'].abs().mean())

                # Calculate direction agreement with generic
                if cond != 'generic':
                    agreements = 0
                    total = 0
                    for _, row in df.iterrows():
                        marker = row['marker']
                        if marker in generic_directions:
                            current_dir = row['cohens_d'] > 0
                            # Check if majority of generic models agree
                            generic_majority = sum(generic_directions[marker]) > len(generic_directions[marker]) / 2
                            if current_dir == generic_majority:
                                agreements += 1
                            total += 1
                    if total > 0:
                        direction_agreements.append(100 * agreements / total)

        mean_sig = np.mean(sig_pcts)
        mean_d = np.mean(mean_ds)

        if cond == 'generic':
            dir_str = "---"
        else:
            dir_str = f"{np.mean(direction_agreements):.0f}%"

        cond_label = {
            'generic': 'Generic improvement',
            'rewrite_only': 'Rewrite-only',
            'voice_preserving': 'Voice-preserving improvement'
        }[cond]

        print(f"{cond_label:<30} {mean_sig:.0f}%{'':10} {mean_d:.2f}{'':8} {dir_str}")


def generate_table3():
    """Table 3: Dimension-level effects under generic improvement prompt.

    Shows: Dimension, Sig/Total and |d| for each model
    """
    print("\n" + "="*60)
    print("TABLE 3: Dimension-level effects under generic improvement prompt")
    print("="*60)

    # Header
    print(f"{'Dimension':<22}", end="")
    for model in MODELS:
        short_name = MODEL_LABELS[model].split()[0]
        print(f" {short_name:>8} {'':<5}", end="")
    print()
    print("-"*70)

    for dim, markers in DIMENSION_MARKERS.items():
        print(f"{dim:<22}", end="")

        for model in MODELS:
            df = load_comparison('generic', model)
            if df is not None:
                dim_data = df[df['marker'].isin(markers)]
                n_total = len(dim_data)
                n_sig = dim_data['significant_fdr'].sum()

                # Mean d with direction indicator
                # Flip sign so positive = rewrite increases (inflation)
                mean_d = -dim_data['cohens_d'].mean()

                if abs(mean_d) > 0.1:
                    arrow = "↑" if mean_d > 0 else "↓"
                else:
                    arrow = ""

                print(f" {n_sig}/{n_total}  {abs(mean_d):.2f}{arrow}", end="")
            else:
                print(f" {'---':>12}", end="")
        print()

    print("\nNote: Arrows indicate direction: ↑ = increase, ↓ = decrease")


def generate_table4():
    """Table 4: Effect sizes across prompt conditions.

    Shows: Model, Generic |d|, Rewrite |d|, Voice-Pres. |d|, Reduction %
    """
    print("\n" + "="*60)
    print("TABLE 4: Effect sizes across prompt conditions")
    print("="*60)
    print(f"{'Model':<25} {'Generic |d|':<12} {'Rewrite |d|':<12} {'Voice-Pres. |d|':<16} {'Reduction':<10}")
    print("-"*75)

    all_generic = []
    all_voice = []

    for model in MODELS:
        generic_d = voice_d = rewrite_d = 0

        for cond in CONDITIONS:
            df = load_comparison(cond, model)
            if df is not None:
                mean_d = df['cohens_d'].abs().mean()
                if cond == 'generic':
                    generic_d = mean_d
                    all_generic.append(mean_d)
                elif cond == 'rewrite_only':
                    rewrite_d = mean_d
                elif cond == 'voice_preserving':
                    voice_d = mean_d
                    all_voice.append(mean_d)

        reduction = 100 * (generic_d - voice_d) / generic_d if generic_d > 0 else 0
        print(f"{MODEL_LABELS[model]:<25} {generic_d:.2f}{'':8} {rewrite_d:.2f}{'':8} {voice_d:.2f}{'':12} {reduction:.0f}%")

    # Mean row
    mean_generic = np.mean(all_generic)
    mean_voice = np.mean(all_voice)
    mean_reduction = 100 * (mean_generic - mean_voice) / mean_generic

    print("-"*75)
    print(f"{'Mean':<25} {mean_generic:.2f}{'':8} {'---':8} {mean_voice:.2f}{'':12} {mean_reduction:.0f}%")


def generate_key_markers():
    """Print key marker statistics mentioned in the paper."""
    print("\n" + "="*60)
    print("KEY MARKERS: Individual marker statistics")
    print("="*60)

    key_markers = [
        ('voice_contraction_density', 'Contraction density'),
        ('ld_mtld', 'MTLD (lexical diversity)'),
        ('voice_first_person_density', 'First-person density'),
    ]

    for marker_id, marker_name in key_markers:
        print(f"\n{marker_name}:")
        print(f"{'Model':<25} {'Generic':<15} {'Voice-Pres.':<15}")
        print("-"*55)

        for model in MODELS:
            generic_pct = voice_pct = "---"

            for cond in ['generic', 'voice_preserving']:
                df = load_comparison(cond, model)
                if df is not None:
                    row = df[df['marker'] == marker_id]
                    if len(row) > 0:
                        pct = row['pct_change'].iloc[0]
                        pct_str = f"{pct:+.0f}%" if not pd.isna(pct) else "---"
                        if cond == 'generic':
                            generic_pct = pct_str
                        else:
                            voice_pct = pct_str

            print(f"{MODEL_LABELS[model]:<25} {generic_pct:<15} {voice_pct:<15}")


def main():
    print("Generating tables from paper results...")
    print("Data directory:", RESULTS_DIR.absolute())

    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_key_markers()

    print("\n" + "="*60)
    print("All tables generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
