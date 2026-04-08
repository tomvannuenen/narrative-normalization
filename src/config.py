"""Configuration for the EmpathicStories LLM rewriting study.

Investigating narrative normalization in LLM-edited personal stories.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# EmpathicStories specific paths
EMPATHIC_STORIES_RAW = RAW_DIR / "empathic_stories_150plus.parquet"

# ── Sampling ───────────────────────────────────────────────────────────
N_SAMPLES = 300               # number of stories to sample from the dataset
MIN_WORD_COUNT = 150          # minimum words to keep a story
MAX_WORD_COUNT = 1000         # personal narratives tend to be shorter than fiction
RANDOM_SEED = 42

# ── LLM rewriting ─────────────────────────────────────────────────────
# Set these via environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

# Updated to latest model versions (March 2026)
REWRITE_MODELS = [
    {
        "provider": "openai",
        "model": "gpt-5.4",              # Latest: GPT-5.4 (released March 5, 2026)
        "label": "gpt54",
    },
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",    # Latest: Claude Sonnet 4.6 (balanced, cost-effective)
        "label": "claude_sonnet",
    },
    {
        "provider": "google",
        "model": "gemini-3.1-pro-preview",  # Gemini 3.1 Pro - using sync mode
        "label": "gemini_31_pro",
    },
]

# ── Prompt Conditions ──────────────────────────────────────────────────
# Testing 3 conditions to measure impact of instruction framing

PROMPT_CONDITIONS = {
    "generic": {
        "prompt": "Please improve the following story while preserving its meaning.\n\n{text}",
        "label": "generic",
        "description": "Generic improvement instruction (baseline)",
    },
    "voice_preserving": {
        "prompt": "Please improve the following story while preserving the author's voice and emotional tone.\n\n{text}",
        "label": "voice",
        "description": "Explicit voice preservation instruction",
    },
    "rewrite_only": {
        "prompt": "Please rewrite the following story.\n\n{text}",
        "label": "rewrite",
        "description": "Neutral rewrite without 'improve' instruction (control)",
    },
}

# Default prompt for single-condition runs
REWRITE_PROMPT = PROMPT_CONDITIONS["generic"]["prompt"]

REWRITE_MAX_TOKENS = 4096
REWRITE_TEMPERATURE = 0.7     # moderate temperature to reflect typical usage

# ── Data files ─────────────────────────────────────────────────────────
# Raw rewrites (may contain preamble/postamble commentary)
REWRITES_RAW = PROCESSED_DIR / "rewrites.parquet"
# Cleaned rewrites (commentary stripped) - USE THIS FOR ANALYSIS
REWRITES_CLEANED = PROCESSED_DIR / "rewrites_cleaned.parquet"

def get_rewrite_column(model_label: str, cleaned: bool = True) -> str:
    """Get the column name for a model's rewrite.

    Args:
        model_label: One of 'gpt52', 'claude_opus', 'gemini_pro'
        cleaned: If True, use cleaned version (recommended for analysis)

    Returns:
        Column name like 'rewrite_gpt52' or 'rewrite_gpt52_clean'
    """
    suffix = "_clean" if cleaned else ""
    return f"rewrite_{model_label}{suffix}"

# ── Linguistic analysis ────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Statistical testing ────────────────────────────────────────────────
ALPHA = 0.05                  # significance level

# ── Marker exclusions ──────────────────────────────────────────────────
# Markers excluded from comparison analysis. Identified via variance audit.

# Floor effects: >85% zero in original stories corpus
EXCLUDED_FLOOR_EFFECTS = [
    "moral_fairness",        # 96% zero
    "moral_authority",       # 98% zero
    "moral_sanctity",        # 100% zero
    "reflective_markers",    # 91% zero
    "conflict_verb_density", # 92% zero
    "growth_markers",        # 88% zero
    "conflict_blame_density",# 87% zero
    "coh_connective_density",# 87% zero — interpret with caution
    "moral_care",            # 84% zero
]

# Redundant/indefensible markers (correlation audit on markers_story.parquet)
EXCLUDED_REDUNDANT = [
    # ── Exact duplicates (r = 1.0) ──────────────────────────────────────
    "sd_mean_consecutive",       # identical to coh_local_similarity
    "sd_mean_pairwise",          # identical to coh_global_similarity
    "pos_verb_ratio",            # identical to narr_action_verb_density (keep narr_)
    "pos_propn_ratio",           # identical to voice_proper_noun_density (keep voice_)
    "sty_punct_ratio",           # r=0.995 with pos_punct_ratio

    # ── pos_x_ratio artifact ─────────────────────────────────────────────
    # spaCy "other" POS tag; r≈0.95 with sent_len, parse_depth — noise
    "pos_x_ratio",
    "pos_intj_ratio",            # r=1.0 with pos_x_ratio

    # ── Raw counts / length proxies ──────────────────────────────────────
    "ld_total_words",            # word count, not a style marker
    "sent_range_compound",       # derived from sent_min − sent_max

    # ── Redundant readability cluster (keep read_flesch_ease, read_coleman_liau) ──
    "read_flesch_kincaid",       # r=0.97 with read_ari
    "read_ari",                  # r=0.94 with read_gunning_fog
    "read_gunning_fog",          # r=0.97 with read_flesch_kincaid

    # ── Redundant LD (keep ld_mtld — length-independent, ld_hapax_ratio) ──
    "ld_ttr",                    # length-dependent; MTLD supersedes
    "ld_root_ttr",               # r=0.955 with ent_word; MTLD supersedes

    # ── Redundant entropy (keep ent_word as information-theoretic measure) ──
    "ent_bigram",                # r=0.969 with ld_unique_words

    # ── Redundant emotion (keep emo_granularity — entropy-based) ─────────
    "emo_distinct_emotions",     # r=0.914 with emo_granularity

    # ── Redundant syntax (keep syn_std_sent_len) ─────────────────────────
    "syn_max_sent_len",          # r=0.921 with syn_std_sent_len

    # ── POS ratios without clear narrative interpretation ────────────────
    "pos_adp_ratio",             # prepositions — no theoretical account
    "pos_det_ratio",             # determiners — no theoretical account
    "pos_aux_ratio",             # auxiliaries — no theoretical account
    "pos_cconj_ratio",           # coordinating conjunctions
    "pos_sconj_ratio",           # subordinating conjunctions (syn_subordination_ratio better)
    "pos_part_ratio",            # particles
    "pos_num_ratio",             # r=0.934 with voice_specific_number_density (keep voice_)
]

EXCLUDED_MARKERS = EXCLUDED_FLOOR_EFFECTS + EXCLUDED_REDUNDANT
