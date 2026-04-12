"""Core stylometric features grounded in authorship attribution research.

This module implements features validated in the stylometric literature,
organized following Stamatatos (2009) taxonomy and the Delta tradition
(Burrows 2002, Eder et al. 2016).

Key Literature:
- Stamatatos (2009): Survey of authorship attribution
- Stamatatos (2013): Robustness of character n-gram features
- Burrows (2002): Delta measure for authorship attribution
- Eder, Rybicki & Kestemont (2016): Stylometry with R (stylo package)
- Mikros (2007): Topic sensitivity of stylometric variables
- Mendenhall (1887): Word-length distribution
- Yule (1944): Statistical Study of Literary Vocabulary
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

import numpy as np
import spacy

from src.config import SPACY_MODEL


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load(SPACY_MODEL)


# ═══════════════════════════════════════════════════════════════════════
# 1. CHARACTER N-GRAMS (Stamatatos 2009, 2013)
# ═══════════════════════════════════════════════════════════════════════
# Character n-grams are the most robust stylometric feature according to
# Stamatatos (2013). They capture morphological patterns, punctuation habits,
# and are relatively topic-invariant.

def character_ngrams(text: str, n_values: tuple[int, ...] = (2, 3, 4)) -> dict:
    """Extract character n-gram frequency distributions.

    Following Stamatatos (2013), we compute character n-grams at the
    character level (including spaces and punctuation), which captures
    word boundaries and morphological patterns.

    Returns entropy and top n-gram features for each n value.
    """
    result = {}

    # Normalize: lowercase, collapse whitespace
    text_norm = re.sub(r'\s+', ' ', text.lower().strip())

    if len(text_norm) < max(n_values):
        for n in n_values:
            result[f'char_{n}gram_entropy'] = 0.0
            result[f'char_{n}gram_hapax_ratio'] = 0.0
            result[f'char_{n}gram_unique_ratio'] = 0.0
        return result

    for n in n_values:
        # Extract all n-grams
        ngrams = [text_norm[i:i+n] for i in range(len(text_norm) - n + 1)]
        if not ngrams:
            result[f'char_{n}gram_entropy'] = 0.0
            result[f'char_{n}gram_hapax_ratio'] = 0.0
            result[f'char_{n}gram_unique_ratio'] = 0.0
            continue

        freq = Counter(ngrams)
        total = sum(freq.values())
        n_types = len(freq)

        # Shannon entropy of n-gram distribution
        probs = [c / total for c in freq.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Hapax ratio (n-grams appearing once)
        hapax = sum(1 for c in freq.values() if c == 1)
        hapax_ratio = hapax / n_types if n_types > 0 else 0.0

        # Type-token ratio for n-grams
        unique_ratio = n_types / total if total > 0 else 0.0

        result[f'char_{n}gram_entropy'] = entropy
        result[f'char_{n}gram_hapax_ratio'] = hapax_ratio
        result[f'char_{n}gram_unique_ratio'] = unique_ratio

    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. MOST FREQUENT WORDS (MFW) - Core of Delta Method
# ═══════════════════════════════════════════════════════════════════════
# Following Burrows (2002) and Eder et al. (2016), function words are
# the most discriminative features for authorship attribution.

# Top function words from stylo/Delta tradition
# Based on Burrows (2002) and expanded by Eder et al. (2016)
TOP_FUNCTION_WORDS = [
    'the', 'and', 'to', 'of', 'a', 'i', 'in', 'was', 'it', 'that',
    'he', 'you', 'his', 'her', 'is', 'had', 'with', 'for', 'she', 'my',
    'not', 'but', 'be', 'at', 'on', 'as', 'have', 'him', 'me', 'all',
    'so', 'we', 'they', 'what', 'this', 'were', 'from', 'would', 'been', 'or',
    'which', 'an', 'when', 'them', 'there', 'no', 'if', 'out', 'by', 'could',
]


def mfw_frequencies(text: str, mfw_list: list[str] = TOP_FUNCTION_WORDS) -> dict:
    """Compute individual frequencies of most frequent words.

    Following the Delta tradition (Burrows 2002), we compute relative
    frequencies of function words, which are largely topic-independent
    and capture individual writing habits.

    Returns normalized frequencies (per 1000 words) for each MFW.
    """
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {f'mfw_{word}': 0.0 for word in mfw_list}

    freq = Counter(tokens)

    # Normalize per 1000 words (following stylo convention)
    result = {}
    for word in mfw_list:
        count = freq.get(word, 0)
        result[f'mfw_{word}'] = (count / n_tokens) * 1000

    return result


def mfw_profile(text: str, n_words: int = 100) -> dict:
    """Compute aggregate MFW statistics.

    Instead of individual word frequencies (which creates many features),
    this computes distributional statistics of the top-N word usage.
    """
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'mfw_top10_coverage': 0.0,
            'mfw_top50_coverage': 0.0,
            'mfw_concentration': 0.0,
            'mfw_fw_ratio': 0.0,
        }

    freq = Counter(tokens)
    sorted_words = freq.most_common()

    # Coverage by top N words
    top10_count = sum(c for _, c in sorted_words[:10])
    top50_count = sum(c for _, c in sorted_words[:50])

    # Concentration: how much the distribution is skewed toward top words
    # Uses Gini coefficient approximation
    counts = np.array([c for _, c in sorted_words])
    if len(counts) > 1:
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n
    else:
        gini = 0.0

    # Function word ratio (how much of the text is function words)
    fw_set = set(TOP_FUNCTION_WORDS)
    fw_count = sum(freq.get(w, 0) for w in fw_set)

    return {
        'mfw_top10_coverage': top10_count / n_tokens,
        'mfw_top50_coverage': top50_count / n_tokens,
        'mfw_concentration': float(gini),
        'mfw_fw_ratio': fw_count / n_tokens,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. VOCABULARY RICHNESS (Beyond TTR/MTLD)
# ═══════════════════════════════════════════════════════════════════════
# Additional vocabulary measures from classical stylometry.
# See Tweedie & Baayen (1998) and Mikros (2007).

def vocabulary_richness(text: str) -> dict:
    """Compute additional vocabulary richness metrics.

    These complement TTR/MTLD with measures that have different
    sensitivity properties:
    - Yule's K: Concentration measure, less sensitive to text length
    - Simpson's D: Probability of two random words being the same
    - Honore's R: Based on hapax legomena, stable across lengths
    - Sichel's S: Based on dis legomena (words appearing twice)
    """
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens < 2:
        return {
            'vocab_yules_k': 0.0,
            'vocab_simpsons_d': 0.0,
            'vocab_honores_r': 0.0,
            'vocab_sichels_s': 0.0,
            'vocab_brunets_w': 0.0,
        }

    freq = Counter(tokens)
    n_types = len(freq)

    # Frequency spectrum: how many words appear exactly m times
    spectrum = Counter(freq.values())

    # V(m) = number of words appearing exactly m times
    V1 = spectrum.get(1, 0)  # hapax legomena
    V2 = spectrum.get(2, 0)  # dis legomena

    # Yule's K (Yule 1944)
    # K = 10^4 * (sum(m^2 * V(m)) - N) / N^2
    sum_m2_vm = sum(m * m * vm for m, vm in spectrum.items())
    yules_k = 10000 * (sum_m2_vm - n_tokens) / (n_tokens ** 2) if n_tokens > 0 else 0.0

    # Simpson's D
    # D = sum(V(m) * m * (m-1)) / (N * (N-1))
    sum_vm_m_m1 = sum(vm * m * (m - 1) for m, vm in spectrum.items())
    simpsons_d = sum_vm_m_m1 / (n_tokens * (n_tokens - 1)) if n_tokens > 1 else 0.0

    # Honore's R (Honore 1979)
    # R = 100 * log(N) / (1 - V1/V)
    if V1 < n_types and n_types > 0:
        honores_r = 100 * math.log(n_tokens) / (1 - V1 / n_types)
    else:
        honores_r = 0.0

    # Sichel's S (based on dis legomena)
    # S = V2 / V
    sichels_s = V2 / n_types if n_types > 0 else 0.0

    # Brunet's W (Brunet 1978)
    # W = N^(V^-0.172)
    if n_types > 0:
        brunets_w = n_tokens ** (n_types ** -0.172)
    else:
        brunets_w = 0.0

    return {
        'vocab_yules_k': yules_k,
        'vocab_simpsons_d': simpsons_d,
        'vocab_honores_r': honores_r,
        'vocab_sichels_s': sichels_s,
        'vocab_brunets_w': brunets_w,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. WORD-LENGTH DISTRIBUTION (Mendenhall 1887)
# ═══════════════════════════════════════════════════════════════════════
# One of the earliest stylometric features. The distribution of word
# lengths is remarkably stable for individual authors.

def word_length_distribution(text: str, max_length: int = 15) -> dict:
    """Compute word-length frequency distribution.

    Following Mendenhall (1887), who first proposed word length as
    a stylometric feature. We compute the proportion of words of
    each length from 1 to max_length.
    """
    doc = _nlp()(text)
    tokens = [t.text for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens == 0:
        result = {f'wlen_{i}': 0.0 for i in range(1, max_length + 1)}
        result['wlen_mean'] = 0.0
        result['wlen_std'] = 0.0
        result['wlen_skew'] = 0.0
        return result

    lengths = [len(t) for t in tokens]
    length_counter = Counter(lengths)

    # Proportion of words of each length
    result = {}
    for i in range(1, max_length + 1):
        result[f'wlen_{i}'] = length_counter.get(i, 0) / n_tokens

    # Aggregate statistics
    lengths_arr = np.array(lengths)
    result['wlen_mean'] = float(lengths_arr.mean())
    result['wlen_std'] = float(lengths_arr.std())

    # Skewness (asymmetry of distribution)
    if lengths_arr.std() > 0:
        skew = float(((lengths_arr - lengths_arr.mean()) ** 3).mean() / (lengths_arr.std() ** 3))
    else:
        skew = 0.0
    result['wlen_skew'] = skew

    return result


# ═══════════════════════════════════════════════════════════════════════
# 5. PUNCTUATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════
# Punctuation usage is highly idiosyncratic and stable.

def punctuation_patterns(text: str) -> dict:
    """Compute punctuation usage patterns.

    Punctuation is a robust stylometric feature because it's
    largely independent of topic and reflects individual habits.
    """
    doc = _nlp()(text)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)
    n_sents = max(len(list(doc.sents)), 1)

    if n_tokens == 0:
        return {
            'punct_comma_ratio': 0.0,
            'punct_semicolon_ratio': 0.0,
            'punct_colon_ratio': 0.0,
            'punct_dash_ratio': 0.0,
            'punct_exclamation_ratio': 0.0,
            'punct_question_ratio': 0.0,
            'punct_quote_ratio': 0.0,
            'punct_ellipsis_ratio': 0.0,
            'punct_total_ratio': 0.0,
            'punct_per_sentence': 0.0,
        }

    # Count specific punctuation marks
    punct_counts = {
        'comma': text.count(','),
        'semicolon': text.count(';'),
        'colon': text.count(':'),
        'dash': text.count('-') + text.count('—') + text.count('–'),
        'exclamation': text.count('!'),
        'question': text.count('?'),
        'quote': text.count('"') + text.count("'") + text.count('"') + text.count('"'),
        'ellipsis': text.count('...') + text.count('…'),
    }

    # Normalize by token count
    result = {}
    total_punct = 0
    for name, count in punct_counts.items():
        result[f'punct_{name}_ratio'] = count / n_tokens
        total_punct += count

    result['punct_total_ratio'] = total_punct / n_tokens
    result['punct_per_sentence'] = total_punct / n_sents

    return result


# ═══════════════════════════════════════════════════════════════════════
# 6. SENTENCE-LENGTH PATTERNS
# ═══════════════════════════════════════════════════════════════════════
# Sentence length distribution is a classic stylometric feature.

def sentence_length_distribution(text: str) -> dict:
    """Compute sentence-length distribution statistics.

    Beyond mean/std, we compute distributional properties that
    characterize the rhythm of writing.
    """
    doc = _nlp()(text)
    sents = list(doc.sents)

    if not sents:
        return {
            'slen_mean': 0.0,
            'slen_std': 0.0,
            'slen_median': 0.0,
            'slen_min': 0.0,
            'slen_max': 0.0,
            'slen_range': 0.0,
            'slen_short_ratio': 0.0,
            'slen_long_ratio': 0.0,
            'slen_variation_coef': 0.0,
        }

    # Count words per sentence (excluding punctuation)
    lengths = [len([t for t in s if t.is_alpha]) for s in sents]
    lengths = [l for l in lengths if l > 0]  # exclude empty sentences

    if not lengths:
        return {
            'slen_mean': 0.0,
            'slen_std': 0.0,
            'slen_median': 0.0,
            'slen_min': 0.0,
            'slen_max': 0.0,
            'slen_range': 0.0,
            'slen_short_ratio': 0.0,
            'slen_long_ratio': 0.0,
            'slen_variation_coef': 0.0,
        }

    arr = np.array(lengths)
    mean = float(arr.mean())
    std = float(arr.std())

    # Short (<10 words) and long (>25 words) sentence ratios
    short_ratio = sum(1 for l in lengths if l < 10) / len(lengths)
    long_ratio = sum(1 for l in lengths if l > 25) / len(lengths)

    # Coefficient of variation (normalized variability)
    cv = std / mean if mean > 0 else 0.0

    return {
        'slen_mean': mean,
        'slen_std': std,
        'slen_median': float(np.median(arr)),
        'slen_min': float(arr.min()),
        'slen_max': float(arr.max()),
        'slen_range': float(arr.max() - arr.min()),
        'slen_short_ratio': short_ratio,
        'slen_long_ratio': long_ratio,
        'slen_variation_coef': cv,
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. BURROWS' DELTA (Stylometric Distance)
# ═══════════════════════════════════════════════════════════════════════
# Delta is the standard distance measure in computational stylometry.
# It measures how different a text's MFW profile is from a reference.

def compute_delta_features(
    text: str,
    reference_texts: Optional[list[str]] = None,
    n_mfw: int = 50,
) -> dict:
    """Compute Burrows' Delta-related features.

    If reference_texts is provided, computes Delta distance to the
    centroid of the reference corpus. Otherwise, computes features
    related to the text's MFW distribution that are relevant for
    Delta-style analysis.

    Following Burrows (2002) and Eder et al. (2016).
    """
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'delta_fw_zscore_mean': 0.0,
            'delta_fw_zscore_std': 0.0,
            'delta_fw_deviation': 0.0,
        }

    freq = Counter(tokens)

    # Compute relative frequencies for function words
    fw_freqs = []
    for word in TOP_FUNCTION_WORDS[:n_mfw]:
        rel_freq = freq.get(word, 0) / n_tokens
        fw_freqs.append(rel_freq)

    fw_arr = np.array(fw_freqs)

    # Without reference corpus, we compute distributional features
    # These capture how "typical" the MFW usage is
    if fw_arr.std() > 0:
        zscores = (fw_arr - fw_arr.mean()) / fw_arr.std()
        zscore_mean = float(np.abs(zscores).mean())
        zscore_std = float(zscores.std())
    else:
        zscore_mean = 0.0
        zscore_std = 0.0

    # Deviation from expected uniform distribution
    expected = 1.0 / n_mfw
    deviation = float(np.sum(np.abs(fw_arr - expected)))

    result = {
        'delta_fw_zscore_mean': zscore_mean,
        'delta_fw_zscore_std': zscore_std,
        'delta_fw_deviation': deviation,
    }

    # If reference corpus provided, compute actual Delta distance
    if reference_texts:
        # Compute corpus statistics
        corpus_freqs = {word: [] for word in TOP_FUNCTION_WORDS[:n_mfw]}

        for ref_text in reference_texts:
            ref_doc = _nlp()(ref_text)
            ref_tokens = [t.text.lower() for t in ref_doc if t.is_alpha]
            ref_n = len(ref_tokens)
            if ref_n == 0:
                continue
            ref_freq = Counter(ref_tokens)
            for word in TOP_FUNCTION_WORDS[:n_mfw]:
                corpus_freqs[word].append(ref_freq.get(word, 0) / ref_n)

        # Compute corpus means and stds
        corpus_means = []
        corpus_stds = []
        for word in TOP_FUNCTION_WORDS[:n_mfw]:
            freqs = corpus_freqs[word]
            if freqs:
                corpus_means.append(np.mean(freqs))
                corpus_stds.append(np.std(freqs) if len(freqs) > 1 else 1.0)
            else:
                corpus_means.append(0.0)
                corpus_stds.append(1.0)

        corpus_means = np.array(corpus_means)
        corpus_stds = np.array(corpus_stds)
        corpus_stds[corpus_stds == 0] = 1.0  # Avoid division by zero

        # Z-score normalization (Burrows' Delta)
        text_zscores = (fw_arr - corpus_means) / corpus_stds

        # Delta = mean absolute deviation from corpus centroid
        delta = float(np.mean(np.abs(text_zscores)))
        result['delta_distance'] = delta

    return result


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

STYLOMETRIC_DIMENSIONS = [
    ("character_ngrams", character_ngrams),
    ("mfw_profile", mfw_profile),
    ("vocabulary_richness", vocabulary_richness),
    ("word_length_distribution", word_length_distribution),
    ("punctuation_patterns", punctuation_patterns),
    ("sentence_length_distribution", sentence_length_distribution),
    ("delta_features", compute_delta_features),
]


def compute_stylometric_markers(text: str, include_mfw_individual: bool = False) -> dict:
    """Compute all core stylometric markers.

    Args:
        text: Input text to analyze
        include_mfw_individual: If True, include individual MFW frequencies
                               (adds 50 features). Default False for efficiency.

    Returns:
        Dictionary of computed stylometric markers
    """
    result = {}

    for name, fn in STYLOMETRIC_DIMENSIONS:
        try:
            result.update(fn(text))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    # Optionally include individual MFW frequencies
    if include_mfw_individual:
        try:
            result.update(mfw_frequencies(text))
        except Exception as e:
            print(f"  Warning: mfw_frequencies failed: {e}")

    return result


def compute_stylometric_delta(
    original: str,
    rewritten: str,
    reference_corpus: Optional[list[str]] = None,
) -> dict:
    """Compute stylometric delta between original and rewritten text.

    This provides a direct measure of how much the stylometric profile
    has changed due to rewriting.

    Returns delta values for key feature categories.
    """
    orig_markers = compute_stylometric_markers(original)
    rewrite_markers = compute_stylometric_markers(rewritten)

    # Compute differences for each marker
    deltas = {}
    for key in orig_markers:
        if key in rewrite_markers:
            diff = rewrite_markers[key] - orig_markers[key]
            deltas[f'{key}_delta'] = diff

    # Aggregate statistics
    delta_values = list(deltas.values())
    if delta_values:
        deltas['stylometric_mean_abs_delta'] = float(np.mean(np.abs(delta_values)))
        deltas['stylometric_max_abs_delta'] = float(np.max(np.abs(delta_values)))

    return deltas
