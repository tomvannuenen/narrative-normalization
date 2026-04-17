"""Compute narrative stance markers for experiential vs explanatory analysis.

This module computes markers that operationalize the distinction between:
- Experiential narration: situated, event-focused, immediate
- Explanatory narration: retrospective, interpretive, distanced

Key marker groups:
1. Epistemic stance and certainty
2. Narratorial involvement and presence
3. Orality vs literariness cues
4. Temporal and causal structure
5. Affective and interpersonal positioning
6. Experiential vs explanatory structure (Section 6)

The main function is `compute_indexical_markers(text)` which returns a flat
dictionary of all computed markers with an ``idx_`` prefix.

For the narrative stance shift analysis, the key markers from Section 6 are:
- idx_eventive_clause_density: proportion of clauses with eventive verbs
- idx_cognitive_clause_density: proportion of clauses with cognitive verbs
- idx_abstraction_density: abstract nouns per 100 words
- idx_retrospective_framing_density: retrospective phrases per sentence
- idx_causal_explanatory_density: causal connectives per sentence
- idx_fp_eventive_density: first-person + eventive verb constructions
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import re
try:
    from src.config import SPACY_MODEL
except ImportError:
    SPACY_MODEL = "en_core_web_sm"  # Default fallback


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=1)
def _phrase_matchers() -> dict[str, PhraseMatcher]:
    """Build lowercase phrase matchers for multiword constructions."""
    nlp = _nlp()

    lexicons = {
        "hedges": [
            "sort of", "kind of", "i think", "i guess", "i suppose",
            "i believe", "i feel like", "it seems", "it appears",
            "not sure", "not certain", "i don't know", "i'm not sure",
        ],
        "boosters": [
            "of course", "without doubt", "no doubt",
        ],
        "certainty": [
            "without question", "no doubt", "of course",
        ],
        "temporal": [
            "as soon as", "at first", "in the end",
        ],
        "causal": [
            "as a result", "due to", "owing to", "for this reason",
            "that's why", "which is why",
        ],
        "evaluative": [
            "the point is", "what matters", "the thing is", "what i mean",
            "in retrospect", "looking back", "in hindsight",
        ],
        "closure": [
            "in the end", "to this day", "ever since", "from then on",
        ],
        "interpersonal": [
            "thank you", "i'm sorry", "i am sorry", "excuse me",
        ],
        "moral": [
            "have to", "need to", "ought to",
        ],
        "discourse_parenthetical": [
            "you know", "i mean",
        ],
    }

    matchers: dict[str, PhraseMatcher] = {}
    for name, phrases in lexicons.items():
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        if phrases:
            matcher.add(name, [nlp.make_doc(p) for p in phrases])
        matchers[name] = matcher
    return matchers


@lru_cache(maxsize=1)
def _matcher() -> Matcher:
    """Token-pattern matcher for constructions with indexical potential."""
    nlp = _nlp()
    matcher = Matcher(nlp.vocab)

    # First-person epistemic parentheticals and complement-taking uses.
    matcher.add(
        "FIRST_PERSON_EPISTEMIC",
        [
            [{"LOWER": {"IN": ["i", "we"]}}, {"LEMMA": {"IN": ["think", "guess", "suppose", "believe", "feel", "assume", "wonder", "doubt"]}}],
            [{"LOWER": "it"}, {"LEMMA": {"IN": ["seem", "appear"]}}],
        ],
    )

    # Sentence-initial discourse markers are more defensible than raw counts.
    matcher.add(
        "SENT_INITIAL_DM",
        [
            [
                {"IS_SENT_START": True},
                {"LOWER": {"IN": ["well", "so", "anyway", "actually", "honestly", "seriously", "okay", "ok", "now", "then"]}},
            ]
        ],
    )

    # Closure patterns with first-person retrospective verbs.
    matcher.add(
        "RETROSPECTIVE_CLOSURE",
        [
            [{"LOWER": {"IN": ["i", "we"]}}, {"LEMMA": {"IN": ["learn", "realize", "understand", "discover"]}}],
        ],
    )

    return matcher


def _alpha_tokens(doc) -> list:
    return [t for t in doc if t.is_alpha]


def _sentences(doc) -> list:
    return list(doc.sents)


def _safe_word_count(doc) -> int:
    return max(len(_alpha_tokens(doc)), 1)


def _safe_sent_count(doc) -> int:
    return max(len(_sentences(doc)), 1)


def _count_phrase_matches(doc, matcher_name: str) -> int:
    matcher = _phrase_matchers()[matcher_name]
    matches = matcher(doc)
    spans = {(start, end) for _, start, end in matches}
    return len(spans)


def _count_token_set(doc, items: Iterable[str]) -> int:
    items = set(items)
    return sum(1 for t in doc if t.is_alpha and t.lower_ in items)


# ═══════════════════════════════════════════════════════════════════════
# 1. EPISTEMIC STANCE AND CERTAINTY
# ═══════════════════════════════════════════════════════════════════════

# Single-token hedges and downtoners. Multiword hedges handled via PhraseMatcher.
HEDGE_TOKENS = {
    "maybe", "perhaps", "possibly", "probably", "apparently", "seemingly",
    "somewhat", "rather", "fairly", "quite", "might", "could", "may",
}

BOOSTER_TOKENS = {
    "really", "very", "absolutely", "completely", "totally", "extremely",
    "incredibly", "definitely", "certainly", "obviously", "clearly",
    "surely", "always", "never", "literally", "truly",
}

CERTAINTY_TOKENS = {
    "clearly", "obviously", "certainly", "definitely", "undoubtedly",
    "naturally",
}

UNCERTAINTY_TOKENS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "may",
}


def epistemic_stance(text: str) -> dict:
    """Measure candidate cues of epistemic positioning.

    These features are best interpreted as resources for stance-taking rather than
    direct measurements of certainty or uncertainty.
    """
    doc = _nlp()(text)
    n_words = _safe_word_count(doc)
    n_sents = _safe_sent_count(doc)

    hedge_count = _count_token_set(doc, HEDGE_TOKENS) + _count_phrase_matches(doc, "hedges")
    booster_count = _count_token_set(doc, BOOSTER_TOKENS) + _count_phrase_matches(doc, "boosters")

    matches = _matcher()(doc)
    epistemic_construction_count = sum(1 for mid, _, _ in matches if doc.vocab.strings[mid] == "FIRST_PERSON_EPISTEMIC")

    certainty_count = _count_token_set(doc, CERTAINTY_TOKENS) + _count_phrase_matches(doc, "certainty")
    uncertainty_count = _count_token_set(doc, UNCERTAINTY_TOKENS) + _count_phrase_matches(doc, "hedges")

    cert_uncert_ratio = (
        certainty_count / uncertainty_count if uncertainty_count > 0
        else (1.0 if certainty_count > 0 else 0.0)
    )

    return {
        "idx_hedge_density": hedge_count / n_sents,
        "idx_booster_density": booster_count / n_sents,
        "idx_epistemic_construction_density": epistemic_construction_count / n_sents,
        "idx_certainty_ratio": certainty_count / n_sents,
        "idx_uncertainty_ratio": uncertainty_count / n_sents,
        "idx_certainty_uncertainty_ratio": cert_uncert_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. NARRATORIAL INVOLVEMENT AND PRESENCE
# ═══════════════════════════════════════════════════════════════════════

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}


def _quote_density(text: str, n_sents: int) -> float:
    # Approximate direct-quotation density using paired quotes only.
    quote_marks = text.count('"')
    paired_quotes = quote_marks // 2
    return paired_quotes / n_sents


def narratorial_involvement(text: str) -> dict:
    """Measure involvement, narrator presence, and reader orientation."""
    doc = _nlp()(text)
    n_words = _safe_word_count(doc)
    n_sents = _safe_sent_count(doc)

    fp_sing_count = _count_token_set(doc, FIRST_PERSON_SINGULAR)
    fp_plural_count = _count_token_set(doc, FIRST_PERSON_PLURAL)
    second_person_count = _count_token_set(doc, SECOND_PERSON)

    matches = _matcher()(doc)
    sent_initial_dm_count = sum(1 for mid, _, _ in matches if doc.vocab.strings[mid] == "SENT_INITIAL_DM")
    parenthetical_dm_count = _count_phrase_matches(doc, "discourse_parenthetical")

    question_count = text.count("?")
    exclamation_count = text.count("!")

    verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
    n_verbs = max(len(verbs), 1)
    present_count = sum(1 for v in verbs if v.tag_ in {"VBP", "VBZ", "VBG"})
    past_count = sum(1 for v in verbs if v.tag_ in {"VBD", "VBN"})

    return {
        "idx_first_person_singular": fp_sing_count / n_words * 100,
        "idx_first_person_plural": fp_plural_count / n_words * 100,
        "idx_second_person": second_person_count / n_words * 100,
        "idx_discourse_marker_density": (sent_initial_dm_count + parenthetical_dm_count) / n_sents,
        "idx_question_density": question_count / n_sents,
        "idx_exclamation_density": exclamation_count / n_sents,
        "idx_present_tense_ratio": present_count / n_verbs,
        "idx_past_tense_ratio": past_count / n_verbs,
        "idx_direct_speech_density": _quote_density(text, n_sents),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. ORALITY VS LITERARINESS
# ═══════════════════════════════════════════════════════════════════════

CONTRACTION_RE = re.compile(
    r"\b(?:[A-Za-z]+n't|[A-Za-z]+'(?:m|re|ve|ll|d|s)|gonna|wanna|gotta|kinda|sorta|dunno|y'all)\b",
    flags=re.IGNORECASE,
)

COORDINATORS = {"and", "but", "or", "so", "yet"}
NOMINALIZATION_SUFFIXES = ("tion", "sion", "ness", "ment", "ity", "ance", "ence")


def orality_literariness(text: str) -> dict:
    """Measure cues associated with spoken-like vs elaborated written texture."""
    doc = _nlp()(text)
    sents = _sentences(doc)
    alpha_tokens = _alpha_tokens(doc)
    n_words = max(len(alpha_tokens), 1)
    n_sents = max(len(sents), 1)

    contraction_count = len(CONTRACTION_RE.findall(text))

    fragment_count = 0
    for sent in sents:
        # Count short, verbless clauses as fragments; ignore sentence-final punctuation artifacts.
        has_finite_or_root_pred = any(
            (t.pos_ in {"VERB", "AUX"} and t.tag_ not in {"VBG", "VBN"}) or t.dep_ == "ROOT"
            for t in sent
        )
        if not has_finite_or_root_pred and len([t for t in sent if not t.is_punct and not t.is_space]) > 1:
            fragment_count += 1

    words = [t.lower_ for t in alpha_tokens]
    repetition_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])

    # Paratactic chaining proxy: coordinator density within sentences.
    parataxis_count = sum(1 for t in doc if t.lower_ in COORDINATORS and t.pos_ == "CCONJ")

    em_dash_count = text.count("—") + text.count("--")
    comma_count = text.count(",")
    semicolon_count = text.count(";")

    nominalization_count = sum(1 for t in alpha_tokens if t.lower_.endswith(NOMINALIZATION_SUFFIXES))
    word_lengths = [len(t.text) for t in alpha_tokens]
    mean_word_length = float(np.mean(word_lengths)) if word_lengths else 0.0
    complex_words = sum(1 for t in alpha_tokens if len(t.text) > 6)

    return {
        "idx_contraction_density": contraction_count / n_words * 100,
        "idx_fragment_ratio": fragment_count / n_sents,
        "idx_repetition_ratio": repetition_count / n_words * 100 if n_words > 1 else 0.0,
        "idx_clause_chaining": parataxis_count / n_sents,
        "idx_em_dash_density": em_dash_count / n_sents,
        "idx_comma_density": comma_count / n_sents,
        "idx_semicolon_density": semicolon_count / n_sents,
        "idx_nominalization_ratio": nominalization_count / n_words * 100,
        "idx_mean_word_length": mean_word_length,
        "idx_complex_word_ratio": complex_words / n_words * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. TEMPORAL AND CAUSAL STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

TEMPORAL_TOKENS = {
    "then", "when", "while", "before", "after", "during", "until", "since",
    "once", "finally", "eventually", "suddenly", "meanwhile", "afterwards",
    "previously", "later", "earlier", "first", "second", "next", "last",
}

CAUSAL_TOKENS = {
    "because", "since", "therefore", "thus", "hence", "consequently",
}

EVALUATIVE_TOKENS = {
    "important", "significant", "interesting", "surprising", "amazing",
    "terrible", "wonderful", "great", "awful", "incredible", "unbelievable",
}

SEQUENCING_TOKENS = {"first", "second", "third", "then", "next", "finally", "lastly"}


def temporal_causal_structure(text: str) -> dict:
    """Measure cues of narrative organization and explanatory framing."""
    doc = _nlp()(text)
    n_sents = _safe_sent_count(doc)

    temporal_count = _count_token_set(doc, TEMPORAL_TOKENS) + _count_phrase_matches(doc, "temporal")
    causal_count = _count_token_set(doc, CAUSAL_TOKENS) + _count_phrase_matches(doc, "causal")
    evaluative_count = _count_token_set(doc, EVALUATIVE_TOKENS) + _count_phrase_matches(doc, "evaluative")
    closure_count = (
        _count_phrase_matches(doc, "closure")
        + sum(1 for mid, _, _ in _matcher()(doc) if doc.vocab.strings[mid] == "RETROSPECTIVE_CLOSURE")
    )
    sequencing_count = _count_token_set(doc, SEQUENCING_TOKENS)

    return {
        "idx_temporal_connective_density": temporal_count / n_sents,
        "idx_causal_connective_density": causal_count / n_sents,
        "idx_evaluative_density": evaluative_count / n_sents,
        "idx_closure_marker_density": closure_count / n_sents,
        "idx_narrative_sequencing": sequencing_count / n_sents,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. AFFECTIVE AND INTERPERSONAL POSITIONING
# ═══════════════════════════════════════════════════════════════════════

POSITIVE_AFFECT = {
    "happy", "glad", "joy", "love", "loved", "loving", "excited", "amazing",
    "wonderful", "great", "good", "nice", "beautiful", "grateful", "thankful",
    "blessed", "lucky", "proud", "hope", "hopeful", "optimistic",
}

NEGATIVE_AFFECT = {
    "sad", "angry", "upset", "frustrated", "annoyed", "depressed", "anxious",
    "worried", "scared", "afraid", "terrible", "awful", "horrible", "hate",
    "hated", "hurt", "pain", "painful", "stressed", "overwhelmed",
}

INTERPERSONAL_TOKENS = {
    "sorry", "thanks", "please", "forgive", "appreciate", "grateful", "blame",
    "fault", "responsible",
}

MORAL_TOKENS = {"should", "must", "right", "wrong", "fair", "unfair", "deserve", "owe"}


def affective_positioning(text: str) -> dict:
    """Measure affective and interpersonal positioning cues."""
    doc = _nlp()(text)
    n_words = _safe_word_count(doc)
    n_sents = _safe_sent_count(doc)

    positive_count = _count_token_set(doc, POSITIVE_AFFECT)
    negative_count = _count_token_set(doc, NEGATIVE_AFFECT)
    total_affect = positive_count + negative_count
    affect_balance = (positive_count - negative_count) / total_affect if total_affect > 0 else 0.0

    interpersonal_count = _count_token_set(doc, INTERPERSONAL_TOKENS) + _count_phrase_matches(doc, "interpersonal")
    moral_count = _count_token_set(doc, MORAL_TOKENS) + _count_phrase_matches(doc, "moral")

    return {
        "idx_positive_affect": positive_count / n_words * 100,
        "idx_negative_affect": negative_count / n_words * 100,
        "idx_affect_balance": affect_balance,
        "idx_interpersonal_density": interpersonal_count / n_sents,
        "idx_moral_framing_density": moral_count / n_sents,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. EXPERIENTIAL VS EXPLANATORY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════
# These markers operationalize the shift from experiential narration
# (situated, event-focused, immediate) to explanatory narration
# (retrospective, interpretive, distanced).

# Eventive verbs: concrete actions, motion, perception, physical activity
# These anchor narrative in situated experience
EVENTIVE_VERBS = {
    # Motion and physical action
    "go", "went", "walk", "walked", "run", "ran", "come", "came", "move", "moved",
    "jump", "jumped", "fall", "fell", "stand", "stood", "sit", "sat", "lie", "lay",
    "turn", "turned", "stop", "stopped", "start", "started", "leave", "left",
    "arrive", "arrived", "enter", "entered", "exit", "exited", "return", "returned",
    "drive", "drove", "fly", "flew", "ride", "rode", "climb", "climbed",
    # Perception and sensation
    "see", "saw", "hear", "heard", "feel", "felt", "smell", "smelled", "taste", "tasted",
    "look", "looked", "watch", "watched", "notice", "noticed", "spot", "spotted",
    # Speech acts (as narrative events)
    "say", "said", "tell", "told", "ask", "asked", "answer", "answered", "call", "called",
    "shout", "shouted", "whisper", "whispered", "scream", "screamed", "cry", "cried",
    # Physical manipulation
    "take", "took", "give", "gave", "hold", "held", "grab", "grabbed", "push", "pushed",
    "pull", "pulled", "throw", "threw", "catch", "caught", "put", "pick", "picked",
    "open", "opened", "close", "closed", "break", "broke", "fix", "fixed",
    # Eating, drinking, bodily
    "eat", "ate", "drink", "drank", "sleep", "slept", "wake", "woke", "die", "died",
}

# Cognitive/interpretive verbs: mental states, inference, understanding
# These mark retrospective sense-making and explanation
COGNITIVE_VERBS = {
    # Realization and understanding
    "realize", "realized", "understand", "understood", "learn", "learned", "learnt",
    "discover", "discovered", "recognize", "recognized", "grasp", "grasped",
    "comprehend", "comprehended", "figure", "figured",
    # Thinking and cognition
    "think", "thought", "know", "knew", "believe", "believed", "consider", "considered",
    "wonder", "wondered", "imagine", "imagined", "assume", "assumed", "suppose", "supposed",
    "expect", "expected", "suspect", "suspected", "doubt", "doubted",
    # Reflection and evaluation
    "reflect", "reflected", "conclude", "concluded", "decide", "decided", "determine", "determined",
    "assess", "assessed", "evaluate", "evaluated", "judge", "judged",
    # Memory and recall
    "remember", "remembered", "recall", "recalled", "forget", "forgot", "remind", "reminded",
}

# Retrospective framing markers: signal temporal/interpretive distance
RETROSPECTIVE_PHRASES = [
    "looking back", "in retrospect", "in hindsight", "now i know", "now i understand",
    "now i realize", "now i see", "i later learned", "i would later",
    "it turned out", "it would turn out", "as it turned out", "little did i know",
    "i didn't know then", "i didn't realize then", "at the time i",
    "what i didn't know", "what i didn't realize", "i came to understand",
    "i came to realize", "i came to see", "i eventually learned",
    "years later", "months later", "weeks later", "days later",
    "only later", "only then", "it wasn't until", "it took me",
]

RETROSPECTIVE_TOKENS = {
    "eventually", "ultimately", "finally", "afterward", "afterwards", "subsequently",
    "retrospectively", "hindsight", "retrospect",
}

# Causal-explanatory connectives (expanded from existing)
CAUSAL_EXPLANATORY_TOKENS = {
    # Causal
    "because", "since", "therefore", "thus", "hence", "consequently", "accordingly",
    # Result/effect
    "so", "thereby", "result", "resulted", "resulting",
    # Explanation
    "explain", "explained", "means", "meant", "imply", "implied", "suggest", "suggested",
    "indicate", "indicated", "show", "showed", "demonstrate", "demonstrated",
    # Reason-giving
    "reason", "cause", "caused", "why",
}

CAUSAL_EXPLANATORY_PHRASES = [
    "as a result", "due to", "owing to", "for this reason", "that's why", "which is why",
    "this meant", "this means", "what this meant", "the reason was", "the reason is",
    "because of this", "as a consequence", "the result was", "this resulted in",
    "this led to", "which led to", "leading to", "this explains", "which explains",
]

# Abstract noun suffixes (for abstraction score)
ABSTRACT_SUFFIXES = (
    "tion", "sion", "ness", "ment", "ity", "ance", "ence",
    "ism", "ship", "hood", "dom", "acy", "ure",
)

# Concrete/physical nouns (for contrast with abstraction)
CONCRETE_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

# Sensory language lexicons for experiential narration
SENSORY_SIGHT = {
    "see", "saw", "seen", "look", "looked", "looking", "watch", "watched", "watching",
    "notice", "noticed", "noticing", "spot", "spotted", "spotting", "glimpse", "glimpsed",
    "stare", "stared", "staring", "gaze", "gazed", "gazing", "glance", "glanced",
    "bright", "dark", "light", "shadow", "color", "colour", "red", "blue", "green",
    "white", "black", "yellow", "golden", "silver", "shine", "shining", "glow", "glowing",
    "flash", "flashing", "sparkle", "sparkling", "dim", "blur", "blurry", "clear",
    "visible", "invisible", "appear", "appeared", "appearing", "disappear", "disappeared",
}

SENSORY_SOUND = {
    "hear", "heard", "hearing", "listen", "listened", "listening", "sound", "sounded",
    "noise", "noisy", "quiet", "silent", "silence", "loud", "soft", "ring", "ringing",
    "rang", "buzz", "buzzing", "click", "clicked", "clicking", "bang", "banged",
    "crash", "crashed", "crashing", "whisper", "whispered", "whispering", "shout",
    "shouted", "shouting", "scream", "screamed", "screaming", "cry", "cried", "crying",
    "laugh", "laughed", "laughing", "voice", "voices", "echo", "echoed", "echoing",
    "murmur", "murmured", "murmuring", "hum", "hummed", "humming", "roar", "roared",
}

SENSORY_TOUCH = {
    "feel", "felt", "feeling", "touch", "touched", "touching", "hold", "held", "holding",
    "grab", "grabbed", "grabbing", "grip", "gripped", "gripping", "squeeze", "squeezed",
    "push", "pushed", "pushing", "pull", "pulled", "pulling", "stroke", "stroked",
    "soft", "hard", "rough", "smooth", "sharp", "dull", "warm", "cold", "hot", "cool",
    "wet", "dry", "sticky", "slippery", "heavy", "light", "tight", "loose", "pressure",
    "pain", "painful", "ache", "ached", "aching", "tingle", "tingled", "tingling",
    "numb", "burn", "burned", "burning", "freeze", "freezing", "frozen", "shiver",
}

SENSORY_TASTE = {
    "taste", "tasted", "tasting", "eat", "ate", "eating", "drink", "drank", "drinking",
    "swallow", "swallowed", "swallowing", "bite", "bit", "biting", "chew", "chewed",
    "sweet", "sour", "bitter", "salty", "spicy", "bland", "savory", "delicious",
    "disgusting", "flavor", "flavour", "yummy", "gross", "fresh", "stale", "rotten",
}

SENSORY_SMELL = {
    "smell", "smelled", "smelling", "sniff", "sniffed", "sniffing", "scent", "scented",
    "odor", "odour", "aroma", "fragrance", "fragrant", "stink", "stank", "stinking",
    "stinky", "fresh", "musty", "pungent", "foul", "sweet", "perfume", "reek", "reeked",
}

# Combined sensory language set
SENSORY_LANGUAGE = SENSORY_SIGHT | SENSORY_SOUND | SENSORY_TOUCH | SENSORY_TASTE | SENSORY_SMELL


def _get_clause_roots(doc) -> list:
    """Extract clause root verbs from document."""
    roots = []
    for token in doc:
        # Main clause roots
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            roots.append(token)
        # Subordinate clause verbs (advcl, relcl, ccomp, xcomp)
        elif token.dep_ in {"advcl", "relcl", "ccomp", "xcomp", "conj"} and token.pos_ == "VERB":
            roots.append(token)
    return roots


def experiential_explanatory(text: str) -> dict:
    """Measure the balance between experiential and explanatory narration.

    Experiential narration is anchored in situated events, concrete actions,
    and immediate perceptions. Explanatory narration involves retrospective
    interpretation, cognitive framing, and causal reasoning about events.

    This operationalizes the "experiential to explanatory" shift that may
    occur when LLMs rewrite personal narratives.
    """
    doc = _nlp()(text)
    n_words = _safe_word_count(doc)
    n_sents = _safe_sent_count(doc)

    # Get clause roots for clause-level analysis
    clause_roots = _get_clause_roots(doc)
    n_clauses = max(len(clause_roots), 1)

    # Count eventive vs cognitive clauses
    eventive_clause_count = 0
    cognitive_clause_count = 0

    for root in clause_roots:
        lemma = root.lemma_.lower()
        if lemma in EVENTIVE_VERBS or root.text.lower() in EVENTIVE_VERBS:
            eventive_clause_count += 1
        elif lemma in COGNITIVE_VERBS or root.text.lower() in COGNITIVE_VERBS:
            cognitive_clause_count += 1

    # Also count all cognitive verbs (not just as clause roots)
    all_cognitive_count = sum(
        1 for t in doc
        if t.pos_ == "VERB" and (t.lemma_.lower() in COGNITIVE_VERBS or t.text.lower() in COGNITIVE_VERBS)
    )

    # Retrospective framing
    text_lower = text.lower()
    retrospective_phrase_count = sum(1 for phrase in RETROSPECTIVE_PHRASES if phrase in text_lower)
    retrospective_token_count = _count_token_set(doc, RETROSPECTIVE_TOKENS)
    total_retrospective = retrospective_phrase_count + retrospective_token_count

    # Causal-explanatory connectives
    causal_token_count = _count_token_set(doc, CAUSAL_EXPLANATORY_TOKENS)
    causal_phrase_count = sum(1 for phrase in CAUSAL_EXPLANATORY_PHRASES if phrase in text_lower)
    total_causal_explanatory = causal_token_count + causal_phrase_count

    # Abstraction score: ratio of abstract nouns to all nouns
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    n_nouns = max(len(nouns), 1)
    abstract_nouns = sum(1 for t in nouns if t.text.lower().endswith(ABSTRACT_SUFFIXES))

    # Concrete noun density: nouns that are NOT abstract
    # Concrete nouns anchor narration in physical, tangible reality
    concrete_nouns = sum(1 for t in nouns if not t.text.lower().endswith(ABSTRACT_SUFFIXES))

    # Sensory language density: words from sensory lexicons
    # Sensory language indexes experiential, embodied narration
    sensory_count = _count_token_set(doc, SENSORY_LANGUAGE)

    # Event-to-interpretation ratio
    event_interp_ratio = (
        eventive_clause_count / cognitive_clause_count
        if cognitive_clause_count > 0
        else (float(eventive_clause_count) if eventive_clause_count > 0 else 1.0)
    )

    # First-person immediacy: I + past tense eventive verbs (situated narration)
    # vs I + cognitive verbs (reflective narration)
    fp_eventive = 0
    fp_cognitive = 0
    for sent in doc.sents:
        tokens = list(sent)
        for i, t in enumerate(tokens):
            if t.lower_ in {"i", "we"}:
                # Look for verb in next 3 tokens
                for j in range(i + 1, min(i + 4, len(tokens))):
                    next_t = tokens[j]
                    if next_t.pos_ == "VERB":
                        if next_t.lemma_.lower() in EVENTIVE_VERBS or next_t.text.lower() in EVENTIVE_VERBS:
                            fp_eventive += 1
                        elif next_t.lemma_.lower() in COGNITIVE_VERBS or next_t.text.lower() in COGNITIVE_VERBS:
                            fp_cognitive += 1
                        break

    fp_immediacy_ratio = (
        fp_eventive / fp_cognitive
        if fp_cognitive > 0
        else (float(fp_eventive) if fp_eventive > 0 else 1.0)
    )

    return {
        # Clause-level measures
        "idx_eventive_clause_density": eventive_clause_count / n_clauses,
        "idx_cognitive_clause_density": cognitive_clause_count / n_clauses,
        "idx_event_interpretation_ratio": event_interp_ratio,

        # Verb-level measures
        "idx_cognitive_verb_density": all_cognitive_count / n_sents,

        # Retrospective framing
        "idx_retrospective_framing_density": total_retrospective / n_sents,

        # Causal-explanatory
        "idx_causal_explanatory_density": total_causal_explanatory / n_sents,

        # Abstraction
        "idx_abstract_noun_ratio": abstract_nouns / n_nouns,
        "idx_abstraction_density": abstract_nouns / n_words * 100,

        # Concrete/sensory (experiential anchoring)
        "idx_concrete_noun_density": concrete_nouns / n_words * 100,
        "idx_sensory_language_density": sensory_count / n_sents,

        # First-person immediacy
        "idx_fp_eventive_density": fp_eventive / n_sents,
        "idx_fp_cognitive_density": fp_cognitive / n_sents,
        "idx_fp_immediacy_ratio": fp_immediacy_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE
# ═══════════════════════════════════════════════════════════════════════

ALL_INDEXICAL_DIMENSIONS = [
    ("epistemic_stance", epistemic_stance),
    ("narratorial_involvement", narratorial_involvement),
    ("orality_literariness", orality_literariness),
    ("temporal_causal_structure", temporal_causal_structure),
    ("affective_positioning", affective_positioning),
    ("experiential_explanatory", experiential_explanatory),
]


def compute_indexical_markers(text: str) -> dict:
    """Compute all candidate indexical cues for a text.

    Returns a flat dictionary of all computed markers with an ``idx_`` prefix.
    """
    result: dict[str, float] = {}
    for name, fn in ALL_INDEXICAL_DIMENSIONS:
        try:
            result.update(fn(text))
        except Exception as exc:
            print(f"  Warning: {name} failed: {exc}")
    return result


def compute_all_indexical_markers_for_texts(texts: list[str]) -> list[dict]:
    """Compute indexical markers for a list of texts with a progress bar."""
    from tqdm import tqdm

    results = []
    for text in tqdm(texts, desc="Computing indexical markers"):
        results.append(compute_indexical_markers(text))
    return results
