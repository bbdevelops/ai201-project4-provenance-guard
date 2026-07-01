"""Confidence scorer — combine the two signals into one calibrated score.

This is the *isolated* scoring component (planning.md §3). It takes the two
standardized signal objects (each ``{"score": float|None, "status": str, ...}``)
and returns one combined confidence in [0, 1] — the system's estimated
probability that the text is AI-generated — plus the attribution band. The Flask
route does no scoring math; this module is the single source of truth for the
bands, so the >=0.70 / <0.40 edges live in exactly one place.

Design choices (planning.md §3 + confirmed M4 decisions):
  - Both signals trusted  -> weighted blend, slight lean toward the semantic
    (LLM) signal (0.6 / 0.4).
  - One signal degraded   -> drop the untrusted signal, score on the survivor
    alone, and cap at FALLBACK_CAP (0.69) so a single signal can never reach the
    "likely AI" band. The spec mandates this for the LLM-fallback case; we apply
    it symmetrically (either survivor) because short text — the usual reason the
    stylometric signal drops out — is exactly the LLM's documented blind spot,
    and the false-positive asymmetry says: never brand a creator AI on a single
    signal's say-so.
  - Both degraded         -> no confidence; attribution "uncertain".
"""

# --- Band thresholds (verbatim from planning.md §3) ---------------------------
BAND_AI_MIN = 0.70  # >= this -> likely_ai
BAND_HUMAN_MAX = 0.40  # <  this -> likely_human; the band between is "uncertain"
FALLBACK_CAP = 0.69  # single-signal verdicts can never enter the likely_ai band

# --- Blend weights (slight lean toward the semantic signal) -------------------
W_LLM, W_STYLO = 0.6, 0.4

# --- Ensemble Detection stretch (Signal 3 present) ----------------------------
# Three-signal weights, semantic still highest (planning.md §Stretch). Used only
# when a perplexity signal is passed in; renormalized over whatever subset is
# usable. The required two-signal path above is untouched.
W3_LLM, W3_STYLO, W3_PPL = 0.5, 0.3, 0.2
# Conflict resolution: if the usable signals' scores span more than this, they
# "strongly disagree" — cap the blend at FALLBACK_CAP so disagreement widens into
# the uncertain band rather than forcing a confident AI verdict (§3 asymmetry).
DISAGREE_SPREAD = 0.40


def attribution_for(confidence):
    """Map a combined confidence (or None) to one of three attribution bands."""
    if confidence is None:
        return "uncertain"
    if confidence >= BAND_AI_MIN:
        return "likely_ai"
    if confidence < BAND_HUMAN_MAX:
        return "likely_human"
    return "uncertain"


def _usable(signal):
    """A signal counts only when it succeeded AND carries a real score."""
    return signal.get("status") == "success" and signal.get("score") is not None


def score_confidence(llm_signal, stylo_signal, perplexity_signal=None):
    """Combine the signal objects into a verdict.

    Returns ``{"confidence": float|None, "attribution": str, "mode": str}``.

    When ``perplexity_signal`` is None (the required two-signal system, Signal 3
    disabled) the original two-signal logic runs unchanged; mode is one of:
      "blended"        — both signals trusted, weighted blend
      "fallback_stylo" — LLM degraded, stylometrics alone, capped at 0.69
      "fallback_llm"   — stylometrics degraded, LLM alone, capped at 0.69
      "degraded"       — neither signal usable, no confidence

    When ``perplexity_signal`` is provided (Ensemble Detection stretch) the
    three-signal path runs instead — see ``_score_ensemble`` — with modes
    "ensemble", "ensemble_conflict", "ensemble_fallback", and "degraded".
    """
    if llm_signal.get("status") == "injection_flagged":
        return {
            "confidence": None,
            "attribution": "uncertain",
            "mode": "injection_rejected",
        }

    if perplexity_signal is not None:
        return _score_ensemble(llm_signal, stylo_signal, perplexity_signal)

    llm_ok = _usable(llm_signal)
    stylo_ok = _usable(stylo_signal)

    if llm_ok and stylo_ok:
        blend = W_LLM * llm_signal["score"] + W_STYLO * stylo_signal["score"]
        confidence = round(blend, 3)
        return {
            "confidence": confidence,
            "attribution": attribution_for(confidence),
            "mode": "blended",
        }

    if stylo_ok:
        confidence = round(min(stylo_signal["score"], FALLBACK_CAP), 3)
        return {
            "confidence": confidence,
            "attribution": attribution_for(confidence),
            "mode": "fallback_stylo",
        }

    if llm_ok:
        confidence = round(min(llm_signal["score"], FALLBACK_CAP), 3)
        return {
            "confidence": confidence,
            "attribution": attribution_for(confidence),
            "mode": "fallback_llm",
        }

    return {"confidence": None, "attribution": "uncertain", "mode": "degraded"}


def _score_ensemble(llm_signal, stylo_signal, perplexity_signal):
    """Three-signal blend with disagreement-widening (planning.md §Stretch).

    Weighted blend with the semantic signal weighted highest
    (``0.5/0.3/0.2``), renormalized over whatever subset is usable:
      - 0 usable -> no confidence (degraded).
      - 1 usable -> that signal alone, capped at FALLBACK_CAP (one signal can
        never reach the likely_ai band — §3 false-positive asymmetry).
      - 2-3 usable -> renormalized weighted blend, then conflict resolution: if
        the usable scores span more than DISAGREE_SPREAD, cap at FALLBACK_CAP so
        strong disagreement widens into the uncertain band.
    """
    # (weight, signal) in priority order; keep only the usable ones.
    candidates = [
        (W3_LLM, llm_signal),
        (W3_STYLO, stylo_signal),
        (W3_PPL, perplexity_signal),
    ]
    usable = [(w, s["score"]) for (w, s) in candidates if _usable(s)]

    if not usable:
        return {"confidence": None, "attribution": "uncertain", "mode": "degraded"}

    if len(usable) == 1:
        score = usable[0][1]
        confidence = round(min(score, FALLBACK_CAP), 3)
        return {
            "confidence": confidence,
            "attribution": attribution_for(confidence),
            "mode": "ensemble_fallback",
        }

    # 2 or 3 usable: renormalize weights over the usable subset, then blend.
    total_w = sum(w for (w, _) in usable)
    blend = sum(w * score for (w, score) in usable) / total_w

    # Conflict resolution: strong disagreement -> widen into the uncertain band.
    scores = [score for (_, score) in usable]
    conflict = (max(scores) - min(scores)) > DISAGREE_SPREAD
    if conflict:
        blend = min(blend, FALLBACK_CAP)

    confidence = round(blend, 3)
    return {
        "confidence": confidence,
        "attribution": attribution_for(confidence),
        "mode": "ensemble_conflict" if conflict else "ensemble",
    }
