"""Narrative-specific markers for personal storytelling and empathic narratives.

Based on literature from narrative psychology, moral foundations theory,
and sociolinguistics. These markers complement the general linguistic markers
in markers.py with dimensions specific to lived experience narratives.

Key Literature:
- Emotional dynamics: Nook et al. (2017), Barrett (2017)
- Moral language: Graham et al. (2009), Haidt & Joseph (2004)
- Authenticity: Pennebaker & King (1999), Newman et al. (2003)
- Conflict: Brown & Levinson (1987), Goffman (1959)
- Narrative structure: Labov & Waletzky (1967), Bruner (1991)
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np
import spacy

from src.config import SPACY_MODEL


# ── Lazy model loading ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _nlp():
    return spacy.load(SPACY_MODEL)


# The 8 Plutchik emotion categories in NRC (excluding valence labels positive/negative)
NRC_EMOTION_CATEGORIES = frozenset([
    'anger', 'anticipation', 'disgust', 'fear',
    'joy', 'sadness', 'surprise', 'trust',
])


@lru_cache(maxsize=1)
def _nrc_lexicon_lookup():
    """Return the full NRC Emotion Lexicon via nrclex (6,468 words).

    NRC Emotion Lexicon: Mohammad & Turney (2013).
    Loaded via the nrclex package which bundles the full lexicon.
    Returns a dict mapping word -> frozenset of emotion categories.
    Only includes the 8 Plutchik emotion dimensions (not valence labels).
    """
    from nrclex import NRCLex
    # Access the bundled lexicon directly (avoids per-text instantiation overhead)
    dummy = NRCLex('dummy')
    lexicon = dummy.__lexicon__
    # Filter to emotion categories only (drop 'positive', 'negative')
    return {
        word: frozenset(cats) & NRC_EMOTION_CATEGORIES
        for word, cats in lexicon.items()
        if frozenset(cats) & NRC_EMOTION_CATEGORIES
    }


# ═══════════════════════════════════════════════════════════════════════
# A. EMOTIONAL DYNAMICS
# ═══════════════════════════════════════════════════════════════════════

def emotional_dynamics(text: str) -> dict:
    """Measure emotional granularity and trajectory volatility.

    Based on:
    - Nook et al. (2017): Emotional granularity and mental health
    - Barrett (2017): How Emotions Are Made
    """
    doc = _nlp()(text)
    sents = list(doc.sents)

    if not sents:
        return {
            'emo_granularity': 0.0,
            'emo_volatility': 0.0,
            'emo_word_density': 0.0,
            'emo_distinct_emotions': 0,
        }

    nrc = _nrc_lexicon_lookup()
    tokens = [t.text.lower() for t in doc if t.is_alpha]

    # Emotion word frequency
    emotion_words = [w for w in tokens if w in nrc]
    emotion_density = len(emotion_words) / len(tokens) if tokens else 0.0

    # Granularity: diversity of specific emotions (not just valence)
    all_emotions = []
    for word in emotion_words:
        all_emotions.extend(nrc[word])

    emotion_counter = Counter(all_emotions)
    distinct_emotions = len(emotion_counter)

    # Emotion entropy (granularity measure — higher = more emotionally diverse)
    if emotion_counter:
        total = sum(emotion_counter.values())
        probs = [count / total for count in emotion_counter.values()]
        granularity = -sum(p * np.log2(p) for p in probs if p > 0)
    else:
        granularity = 0.0

    # Trajectory volatility: std of per-sentence emotion word counts
    sent_emotions = []
    for sent in sents:
        sent_tokens = [t.text.lower() for t in sent if t.is_alpha]
        sent_emo_count = sum(1 for w in sent_tokens if w in nrc)
        sent_emotions.append(sent_emo_count)

    volatility = float(np.std(sent_emotions)) if len(sent_emotions) > 1 else 0.0

    return {
        'emo_granularity': granularity,
        'emo_volatility': volatility,
        'emo_word_density': emotion_density,
        'emo_distinct_emotions': distinct_emotions,
    }


# ═══════════════════════════════════════════════════════════════════════
# B. MORAL & REFLECTIVE LANGUAGE
# ═══════════════════════════════════════════════════════════════════════

# Moral Foundations categories (Graham et al., 2009)
MORAL_FOUNDATIONS = {
    'care': ['care', 'caring', 'compassion', 'empathy', 'kindness', 'cruel', 'harm', 'hurt'],
    'fairness': ['fair', 'unfair', 'justice', 'injustice', 'equal', 'rights', 'cheat'],
    'loyalty': ['loyal', 'betray', 'team', 'group', 'together', 'unity'],
    'authority': ['respect', 'obey', 'tradition', 'authority', 'rebel', 'subversion'],
    'sanctity': ['sacred', 'purity', 'pure', 'disgust', 'degradation', 'sin'],
}

REFLECTIVE_MARKERS = [
    r'\bi\s+learned\b',
    r'\bi\s+realized\b',
    r'\bthis\s+taught\s+me\b',
    r'\bi\s+discovered\b',
    r'\bi\s+understood\b',
    r'\bnow\s+i\s+know\b',
    r'\bi\s+came\s+to\s+understand\b',
    r'\blooking\s+back\b',
    r'\bin\s+hindsight\b',
]

CAUSAL_MARKERS = ['because', 'therefore', 'thus', 'consequently', 'as a result', 'so']

GROWTH_MARKERS = [
    r'\bi\s+grew\b',
    r'\bi\s+changed\b',
    r'\bi\s+became\b',
    r'\bit\s+made\s+me\b',
    r'\bi\s+am\s+now\b',
]


def moral_reflective_language(text: str) -> dict:
    """Measure moral foundations and reflective/growth language.

    Based on:
    - Graham et al. (2009): Moral Foundations Theory
    - Adler et al. (2016): Narrative identity and growth
    """
    text_lower = text.lower()
    doc = _nlp()(text)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'moral_care': 0.0,
            'moral_fairness': 0.0,
            'moral_loyalty': 0.0,
            'moral_authority': 0.0,
            'moral_sanctity': 0.0,
            'reflective_markers': 0,
            'causal_markers': 0,
            'growth_markers': 0,
        }

    # Moral foundations word counts
    moral_counts = {}
    for foundation, words in MORAL_FOUNDATIONS.items():
        count = sum(tokens.count(w) for w in words)
        moral_counts[f'moral_{foundation}'] = count / n_tokens

    # Reflective markers — normalized per 100 words to control for length
    reflective_count = sum(1 for pattern in REFLECTIVE_MARKERS if re.search(pattern, text_lower))
    reflective_per100 = reflective_count / n_tokens * 100

    # Causal reasoning markers — normalized per 100 words
    causal_count = sum(text_lower.count(marker) for marker in CAUSAL_MARKERS)
    causal_per100 = causal_count / n_tokens * 100

    # Growth narrative markers — normalized per 100 words
    growth_count = sum(1 for pattern in GROWTH_MARKERS if re.search(pattern, text_lower))
    growth_per100 = growth_count / n_tokens * 100

    return {
        **moral_counts,
        'reflective_markers': reflective_per100,
        'causal_markers': causal_per100,
        'growth_markers': growth_per100,
    }


# ═══════════════════════════════════════════════════════════════════════
# C. AUTHENTICITY & VOICE MARKERS
# ═══════════════════════════════════════════════════════════════════════

DISFLUENCY_MARKERS = ['um', 'uh', 'like', 'you know', 'i mean', 'kind of', 'sort of']

def authenticity_voice(text: str) -> dict:
    """Measure markers of personal voice and authentic expression.

    Based on:
    - Pennebaker & King (1999): Linguistic Inquiry and Word Count
    - Newman et al. (2003): Lying words (deception detection)
    """
    doc = _nlp()(text)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'voice_first_person_density': 0.0,
            'voice_disfluency': 0.0,
            'voice_contraction_density': 0.0,
            'voice_proper_noun_density': 0.0,
            'voice_specific_number_density': 0.0,
        }

    # First-person pronouns
    first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours']
    fp_count = sum(1 for t in tokens if t.text.lower() in first_person)
    fp_density = fp_count / n_tokens

    # Disfluency markers
    text_lower = text.lower()
    disfluency_count = sum(text_lower.count(marker) for marker in DISFLUENCY_MARKERS)
    disfluency_density = disfluency_count / len(text.split())

    # Contractions (informal language)
    contraction_pattern = r"n't|'re|'ve|'ll|'d|'m|'s"
    contractions = len(re.findall(contraction_pattern, text))
    contraction_density = contractions / n_tokens

    # Proper nouns (personal details)
    proper_nouns = sum(1 for t in tokens if t.pos_ == 'PROPN')
    proper_noun_density = proper_nouns / n_tokens

    # Specific numbers and dates
    numbers = sum(1 for t in tokens if t.pos_ == 'NUM' or t.like_num)
    number_density = numbers / n_tokens

    return {
        'voice_first_person_density': fp_density,
        'voice_disfluency': disfluency_density,
        'voice_contraction_density': contraction_density,
        'voice_proper_noun_density': proper_noun_density,
        'voice_specific_number_density': number_density,
    }


# ═══════════════════════════════════════════════════════════════════════
# D. CONFLICT & BLAME LANGUAGE
# ═══════════════════════════════════════════════════════════════════════

CONFLICT_VERBS = [
    'argue', 'argued', 'fight', 'fought', 'disagree', 'disagreed',
    'yell', 'yelled', 'shout', 'shouted', 'scream', 'screamed',
    'confront', 'confronted', 'attack', 'attacked',
]

NEGATIVE_ATTRIBUTION = [
    'blame', 'blamed', 'fault', 'wrong', 'mistake',
    'should have', 'shouldnt', "shouldn't",
]

def conflict_blame_language(text: str) -> dict:
    """Measure conflict intensity and blame attribution.

    Based on:
    - Goffman (1959): Presentation of Self
    - Brown & Levinson (1987): Politeness Theory
    """
    doc = _nlp()(text)
    text_lower = text.lower()
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'conflict_verb_density': 0.0,
            'conflict_blame_density': 0.0,
            'conflict_second_person': 0.0,
        }

    # Conflict verbs
    conflict_count = sum(tokens.count(v) for v in CONFLICT_VERBS)
    conflict_density = conflict_count / n_tokens

    # Blame/attribution language
    blame_count = sum(text_lower.count(phrase) for phrase in NEGATIVE_ATTRIBUTION)
    blame_density = blame_count / len(text.split())

    # Second-person pronouns (you, your) - potential for blame attribution
    second_person = ['you', 'your', 'yours', 'yourself']
    sp_count = sum(1 for t in doc if t.text.lower() in second_person)
    sp_density = sp_count / n_tokens

    return {
        'conflict_verb_density': conflict_density,
        'conflict_blame_density': blame_density,
        'conflict_second_person': sp_density,
    }


# ═══════════════════════════════════════════════════════════════════════
# E. NARRATIVE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

TEMPORAL_MARKERS = [
    'when', 'before', 'after', 'then', 'next', 'later',
    'first', 'second', 'finally', 'eventually', 'meanwhile',
    'yesterday', 'today', 'tomorrow', 'now',
]

def narrative_structure(text: str) -> dict:
    """Measure narrative event structure and temporal organization.

    Based on:
    - Labov & Waletzky (1967): Narrative structure
    - Bruner (1991): Narrative construction of reality
    """
    doc = _nlp()(text)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {
            'narr_action_verb_density': 0.0,
            'narr_temporal_marker_density': 0.0,
            'narr_past_tense_ratio': 0.0,
        }

    # Action verbs (VERB POS tag, excluding auxiliaries)
    verbs = [t for t in tokens if t.pos_ == 'VERB']
    action_verbs = [v for v in verbs if v.dep_ not in ('aux', 'auxpass')]
    action_density = len(action_verbs) / n_tokens

    # Temporal markers
    text_lower = text.lower().split()
    temporal_count = sum(text_lower.count(marker) for marker in TEMPORAL_MARKERS)
    temporal_density = temporal_count / len(text_lower) if text_lower else 0.0

    # Past tense ratio (narratives often in past tense)
    if verbs:
        past_tense = sum(1 for v in verbs if v.tag_ in ('VBD', 'VBN'))
        past_ratio = past_tense / len(verbs)
    else:
        past_ratio = 0.0

    return {
        'narr_action_verb_density': action_density,
        'narr_temporal_marker_density': temporal_density,
        'narr_past_tense_ratio': past_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE
# ═══════════════════════════════════════════════════════════════════════

EMPATHIC_DIMENSIONS = [
    ("emotional_dynamics", emotional_dynamics),
    ("moral_reflective", moral_reflective_language),
    ("authenticity_voice", authenticity_voice),
    ("conflict_blame", conflict_blame_language),
    ("narrative_structure", narrative_structure),
]


def compute_empathic_markers(text: str) -> dict:
    """Run all empathic narrative dimension functions and merge into one flat dict."""
    result = {}
    for name, fn in EMPATHIC_DIMENSIONS:
        try:
            result.update(fn(text))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# SEMANTIC VOICE DISTANCE
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_distances(
    originals: list[str],
    rewrites: list[str],
) -> list[float]:
    """Compute cosine similarity between each original/rewrite pair.

    Uses sentence-transformers (all-MiniLM-L6-v2) to embed both texts,
    then computes cosine similarity. Higher = more similar (voice preserved).
    Lower = more divergent (voice eroded).

    Returns a list of float similarities, one per pair. None for invalid pairs.

    Based on:
    - Reimers & Gurevych (2019): Sentence-BERT
    """
    import numpy as np

    model = _sentence_model()

    # Batch encode all texts at once for efficiency
    all_texts = originals + rewrites
    embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=False)

    n = len(originals)
    orig_embs = embeddings[:n]
    rewrite_embs = embeddings[n:]

    similarities = []
    for o_emb, r_emb in zip(orig_embs, rewrite_embs):
        # Cosine similarity
        norm_o = np.linalg.norm(o_emb)
        norm_r = np.linalg.norm(r_emb)
        if norm_o == 0 or norm_r == 0:
            similarities.append(None)
        else:
            sim = float(np.dot(o_emb, r_emb) / (norm_o * norm_r))
            similarities.append(sim)

    return similarities
