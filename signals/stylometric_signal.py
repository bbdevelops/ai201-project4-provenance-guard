"""Signal 2 — Stylometric heuristics (structural, pure Python).

Where Signal 1 reads the text *semantically*, this signal measures *structure* —
the statistical shape of the writing — using only the standard library (``re``,
``math``). The two signals fail on different inputs, which is what makes the
combination informative (planning.md §2).

Four metrics are computed and each mapped to a per-metric "AI-ness" sub-score in
[0, 1] (higher = more AI-like = more uniform), then blended into one score:

  - Burstiness        — std-dev of sentence lengths. AI paces evenly (low std).
  - Windowed TTR      — vocabulary diversity over a sliding window. AI repeats
                        "safe" vocabulary (low diversity). Windowed, not raw,
                        because raw TTR is confounded by length.
  - Punctuation density — marks per word. AI is moderate-and-regular; both very
                        sparse and very heavy punctuation read human (two-sided).
  - Complexity        — mean sentence length + clause density. AI builds long,
                        evenly multi-clause sentences.

An optional ``content_type`` routes the metrics to a genre baseline profile so
naturally-uniform genres (academic prose, repetition-heavy poetry) are judged
against their own norms rather than generic prose — the same scoring logic with
different reference points in (planning.md §2 mitigation).

Standardized contract returned to the scorer (mirrors Signal 1):
    {"score": float|None, "status": str, "metrics": dict, "genre": str}
where status is one of: "success" | "parse_error".
"""

import math
import re

# --- Status constants (same vocabulary as Signal 1) ---------------------------
STATUS_SUCCESS = "success"
STATUS_PARSE_ERROR = "parse_error"

# --- Tokenization (regex only, no nltk) ---------------------------------------
_SENT_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")  # split on terminal-punctuation runs
_WORD_RE = re.compile(r"[A-Za-z']+")  # words = letter runs, apostrophes kept
_PUNCT_RE = re.compile(r"[,;:\"()\-—?!.]")  # counted punctuation marks
# Clause boundaries: internal punctuation + coordinating/subordinating words.
_CLAUSE_RE = re.compile(
    r"[,;:]|\b(?:and|but|or|because|which|that|while|although|however|"
    r"furthermore|moreover|therefore|thus)\b",
    re.IGNORECASE,
)

# --- Degenerate-input thresholds ----------------------------------------------
MIN_WORDS = 20  # below this the statistics are noise
MIN_SENTENCES = 2  # need >=2 sentences for variance to exist
TTR_WINDOW = 50  # sliding-window size for windowed TTR

# --- Blend weights (sum to 1.0; calibrated against the M4 test inputs) ---------
# Burstiness + complexity carry the most weight (strongest, least-confounded AI
# tells); punctuation is noisiest, so it gets the least.
W_BURST, W_COMPLEX, W_TTR, W_PUNCT = 0.35, 0.25, 0.25, 0.15
# Short-text rule: with W < TTR_WINDOW the windowed TTR is unreliable, so its
# weight is redistributed onto the two strongest metrics.
W_BURST_SHORT, W_COMPLEX_SHORT, W_PUNCT_SHORT = 0.45, 0.35, 0.20

# --- Genre baseline profiles --------------------------------------------------
# Each profile supplies the ramp endpoints / punctuation center used to map raw
# metrics to sub-scores. The SCORING LOGIC never changes — only these reference
# points swap ("different baselines in, same logic"). academic/poetry widen the
# ramps so naturally-uniform genres must be *more* extreme before flagging,
# reducing false positives against genuine creators (planning.md §2).
GENRE_PROFILES = {
    "prose": {
        "burst": (2.0, 9.0),
        "ttr": (0.45, 0.75),
        "len": (12.0, 26.0),
        "clause": (0.5, 2.0),
        "punct_center": 0.13,
        "punct_halfwidth": 0.10,
    },
    "academic": {
        "burst": (1.0, 7.0),
        "ttr": (0.40, 0.70),
        "len": (16.0, 32.0),
        "clause": (0.8, 2.5),
        "punct_center": 0.15,
        "punct_halfwidth": 0.12,
    },
    "poetry": {
        "burst": (0.5, 6.0),
        "ttr": (0.35, 0.65),
        "len": (4.0, 18.0),
        "clause": (0.2, 1.5),
        "punct_center": 0.10,
        "punct_halfwidth": 0.14,
    },
}
DEFAULT_GENRE = "prose"


def _ramp(x, lo, hi):
    """Piecewise-linear map of ``x`` onto [0, 1]: 0 at/below lo, 1 at/above hi."""
    if lo == hi:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _resolve_profile(content_type):
    """Return (genre_name, profile). Unknown/absent tag -> neutral default."""
    genre = (content_type or "").strip().lower()
    if genre in GENRE_PROFILES:
        return genre, GENRE_PROFILES[genre]
    return DEFAULT_GENRE, GENRE_PROFILES[DEFAULT_GENRE]


def _split_sentences(text):
    """Split into sentences, keeping a trailing non-terminated fragment."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _windowed_ttr(words, window=TTR_WINDOW):
    """Mean type-token ratio over sliding windows; whole-text TTR if too short.

    Raw TTR falls as text lengthens (more chances to repeat), so comparing raw
    TTR across inputs of different lengths is unfair. Averaging TTR over
    fixed-size windows removes that length dependence.
    """
    n = len(words)
    if n == 0:
        return 0.0
    if n < window:
        return len(set(words)) / n
    ratios = []
    for start in range(0, n - window + 1):
        chunk = words[start : start + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


def _stdev(values):
    """Population standard deviation (0.0 for fewer than two values)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def analyze_stylometrics(text, content_type=None):
    """Score ``text`` on structural uniformity. See module docstring for contract.

    ``content_type`` is an optional genre hint ("prose"/"academic"/"poetry");
    an absent or unknown value falls back to the neutral ``prose`` profile.
    """
    genre, profile = _resolve_profile(content_type)

    if not text or not text.strip():
        return {
            "score": None,
            "status": STATUS_PARSE_ERROR,
            "metrics": {"detail": "empty_text"},
            "genre": genre,
        }

    sentences = _split_sentences(text)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    total_words = len(words)
    n_sentences = len(sentences)

    # Degenerate input: too little text for stable statistics.
    if total_words < MIN_WORDS or n_sentences < MIN_SENTENCES:
        return {
            "score": None,
            "status": STATUS_PARSE_ERROR,
            "metrics": {
                "detail": "insufficient_text",
                "total_words": total_words,
                "n_sentences": n_sentences,
            },
            "genre": genre,
        }

    # --- Raw metrics ----------------------------------------------------------
    sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    burstiness = _stdev(sent_lengths)
    windowed_ttr = _windowed_ttr(words)
    punct_count = len(_PUNCT_RE.findall(text))
    punct_density = punct_count / total_words
    mean_sentence_len = total_words / n_sentences
    clause_density = len(_CLAUSE_RE.findall(text)) / n_sentences

    # --- Raw -> per-metric AI-ness sub-scores (using the genre profile) -------
    burst_ai = 1.0 - _ramp(burstiness, *profile["burst"])
    ttr_ai = 1.0 - _ramp(windowed_ttr, *profile["ttr"])
    punct_ai = 1.0 - min(
        1.0, abs(punct_density - profile["punct_center"]) / profile["punct_halfwidth"]
    )
    complex_ai = 0.5 * _ramp(mean_sentence_len, *profile["len"]) + 0.5 * _ramp(
        clause_density, *profile["clause"]
    )

    # --- Combine --------------------------------------------------------------
    if total_words < TTR_WINDOW:
        # Windowed TTR is unreliable here; drop it and reweight.
        score = (
            W_BURST_SHORT * burst_ai
            + W_COMPLEX_SHORT * complex_ai
            + W_PUNCT_SHORT * punct_ai
        )
    else:
        score = (
            W_BURST * burst_ai
            + W_COMPLEX * complex_ai
            + W_TTR * ttr_ai
            + W_PUNCT * punct_ai
        )

    return {
        "score": round(score, 3),
        "status": STATUS_SUCCESS,
        "genre": genre,
        "metrics": {
            # Raw measurements (for transparency / debugging / audit telemetry).
            "burstiness": round(burstiness, 3),
            "windowed_ttr": round(windowed_ttr, 3),
            "punct_density": round(punct_density, 3),
            "mean_sentence_len": round(mean_sentence_len, 3),
            "clause_density": round(clause_density, 3),
            "total_words": total_words,
            "n_sentences": n_sentences,
            # Per-metric sub-scores that feed the blend.
            "burst_ai": round(burst_ai, 3),
            "ttr_ai": round(ttr_ai, 3),
            "punct_ai": round(punct_ai, 3),
            "complex_ai": round(complex_ai, 3),
        },
    }
