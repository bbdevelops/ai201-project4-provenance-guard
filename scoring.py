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


def score_confidence(llm_signal, stylo_signal):
    """Combine the two signal objects into a verdict.

    Returns ``{"confidence": float|None, "attribution": str, "mode": str}``
    where mode is one of:
      "blended"        — both signals trusted, weighted blend
      "fallback_stylo" — LLM degraded, stylometrics alone, capped at 0.69
      "fallback_llm"   — stylometrics degraded, LLM alone, capped at 0.69
      "degraded"       — neither signal usable, no confidence
    """
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
