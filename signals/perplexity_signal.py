"""Signal 3 — GPT-2 Small perplexity (statistical language modeling).

Ensemble Detection stretch (planning.md §Stretch Features). Where Signal 1 reads
the text *semantically* and Signal 2 measures its *structure*, this signal
measures how *predictable* the text is to a small language model — a property
neither of the other two owns. GPT-2 Small computes the mean token
log-likelihood of the text; ``perplexity = exp(cross-entropy loss)``. AI prose is
more predictable (low perplexity); human writing reaches for odd word choices
(high perplexity).

It is **opt-in and dependency-light** so the required two-signal system never
depends on ``torch``/``transformers`` (README "Future Work", planning.md):

  - Flag gate first: if ``ENABLE_PERPLEXITY_SIGNAL`` is off, return ``disabled``
    *before importing torch* — the heavy libs are never touched by default.
  - Lazy import + cached model: ``torch``/``transformers`` and the GPT-2 weights
    load only on first real use, then are cached module-level for reuse. Any
    import/load failure degrades cleanly to ``unavailable`` (never crashes
    /submit).

Standardized contract returned to the scorer (mirrors Signals 1 & 2):
    {"score": float|None, "status": str, "metrics": dict}
where status is one of: "success" | "parse_error" | "disabled" | "unavailable".
"""

import math
import re

import config

# --- Status constants (same vocabulary as Signals 1 & 2, plus two stretch states) -
STATUS_SUCCESS = "success"
STATUS_PARSE_ERROR = "parse_error"
STATUS_DISABLED = "disabled"  # ENABLE_PERPLEXITY_SIGNAL is off (default)
STATUS_UNAVAILABLE = "unavailable"  # torch/transformers/model not loadable

# --- Degenerate-input threshold (mirrors Signal 2's MIN_WORDS) -----------------
MIN_WORDS = 20  # below this, perplexity over a few tokens is noise
_WORD_RE = re.compile(r"[A-Za-z']+")

# --- Bounded context -----------------------------------------------------------
MAX_TOKENS = 1024  # GPT-2's context window; longer inputs are truncated

# --- Perplexity -> AI-likelihood mapping (UNCALIBRATED heuristic endpoints) -----
# Low perplexity (predictable) => AI-like => high score; high perplexity => human.
# These endpoints are *reasoned, not fit to labeled ground truth* — documented as
# such in the README. score = 1 - ramp(ppl, PPL_AI, PPL_HUMAN).
PPL_AI = 25.0  # at/below this, fully AI-like -> score 1.0
PPL_HUMAN = 100.0  # at/above this, fully human-like -> score 0.0

# --- Lazy model cache (populated on first successful load) ---------------------
_MODEL = None
_TOKENIZER = None
_LOAD_FAILED = False  # remember a failed load so we don't retry every request


def _ramp(x, lo, hi):
    """Piecewise-linear map of ``x`` onto [0, 1]: 0 at/below lo, 1 at/above hi."""
    if lo == hi:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _load_model():
    """Lazily import torch/transformers and load GPT-2 Small, caching the result.

    Returns (model, tokenizer) on success or (None, None) if anything fails. The
    heavy imports live *inside* this function so the default (flag-off) path never
    pays for them.
    """
    global _MODEL, _TOKENIZER, _LOAD_FAILED
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER
    if _LOAD_FAILED:
        return None, None
    try:
        import torch  # noqa: F401  (imported for side-effect availability check)
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained(config.PERPLEXITY_MODEL)
        model = GPT2LMHeadModel.from_pretrained(config.PERPLEXITY_MODEL)
        model.eval()
        _MODEL, _TOKENIZER = model, tokenizer
        return _MODEL, _TOKENIZER
    except Exception:
        # ImportError (libs absent) or any load/download failure: degrade cleanly.
        _LOAD_FAILED = True
        return None, None


def analyze_perplexity(text):
    """Score ``text`` by GPT-2 Small perplexity. See module docstring for contract.

    Returns ``{"score": float|None, "status": str, "metrics": dict}``.
    """
    # 1. Flag gate FIRST — never import torch when the stretch is disabled.
    if not config.ENABLE_PERPLEXITY_SIGNAL:
        return {"score": None, "status": STATUS_DISABLED, "metrics": {}}

    # 2. Short-text guard — perplexity over a handful of tokens is noise.
    n_words = len(_WORD_RE.findall(text or ""))
    if n_words < MIN_WORDS:
        return {
            "score": None,
            "status": STATUS_PARSE_ERROR,
            "metrics": {"detail": "insufficient_text", "total_words": n_words},
        }

    # 3. Lazy-load the model; degrade cleanly if torch/transformers/model absent.
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        return {
            "score": None,
            "status": STATUS_UNAVAILABLE,
            "metrics": {"detail": "model_unavailable"},
        }

    # 4. Compute perplexity = exp(mean token cross-entropy loss).
    try:
        import torch

        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
        )
        input_ids = enc["input_ids"]
        n_tokens = int(input_ids.shape[1])
        if n_tokens < 2:  # need >=2 tokens for a next-token loss to exist
            return {
                "score": None,
                "status": STATUS_PARSE_ERROR,
                "metrics": {"detail": "insufficient_tokens", "n_tokens": n_tokens},
            }
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        loss = float(out.loss)  # mean cross-entropy (natural log) per token
        perplexity = math.exp(loss)
    except Exception:
        return {
            "score": None,
            "status": STATUS_UNAVAILABLE,
            "metrics": {"detail": "inference_failed"},
        }

    # 5. Map perplexity -> AI-likelihood. Low perplexity => predictable => AI-like.
    score = 1.0 - _ramp(perplexity, PPL_AI, PPL_HUMAN)
    return {
        "score": round(score, 3),
        "status": STATUS_SUCCESS,
        "metrics": {
            "perplexity": round(perplexity, 3),
            "mean_log_likelihood": round(-loss, 3),
            "n_tokens": n_tokens,
        },
    }
