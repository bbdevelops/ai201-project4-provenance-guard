"""Standalone test harness for the confidence scorer (planning.md §M4 verify).

Feeds synthetic ``{score, status}`` signal pairs into score_confidence() so the
band edges and the single-signal 0.69 cap can be checked deterministically,
WITHOUT calling the Groq API. Also asserts the thresholds match planning.md §3
verbatim (>=0.70 likely_ai, <0.40 likely_human, 0.69 fallback cap) — AI tools
silently drift these, so they are re-asserted here.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\test_scoring.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import (  # noqa: E402
    BAND_AI_MIN,
    BAND_HUMAN_MAX,
    DISAGREE_SPREAD,
    FALLBACK_CAP,
    score_confidence,
)


def _llm(score, status="success"):
    return {"score": score, "status": status, "rationale": None, "marker": None}


def _stylo(score, status="success"):
    return {"score": score, "status": status, "metrics": {}, "genre": "prose"}


def _ppl(score, status="success"):
    return {"score": score, "status": status, "metrics": {}}


# (name, llm_signal, stylo_signal, expected_confidence, expected_attribution, expected_mode)
CASES = [
    # --- Both signals trusted: weighted blend (0.6 LLM + 0.4 stylo) -----------
    ("both_high", _llm(0.9), _stylo(0.8), 0.86, "likely_ai", "blended"),
    ("both_low", _llm(0.1), _stylo(0.2), 0.14, "likely_human", "blended"),
    ("both_mid", _llm(0.5), _stylo(0.5), 0.5, "uncertain", "blended"),
    # Signals disagree -> blend lands in the wide uncertain band (the point).
    ("disagree", _llm(0.8), _stylo(0.2), 0.56, "uncertain", "blended"),
    # --- LLM degraded: stylo-only fallback, capped at 0.69 -------------------
    ("llm_parse_error_high_stylo", _llm(None, "parse_error"), _stylo(0.95),
     FALLBACK_CAP, "uncertain", "fallback_stylo"),
    ("llm_injection_high_stylo", _llm(None, "injection_flagged"), _stylo(0.95),
     FALLBACK_CAP, "uncertain", "fallback_stylo"),
    ("llm_parse_error_low_stylo", _llm(None, "parse_error"), _stylo(0.2),
     0.2, "likely_human", "fallback_stylo"),
    # --- Stylo degraded: LLM-only fallback, capped at 0.69 (symmetric) -------
    ("stylo_parse_error_high_llm", _llm(0.95), _stylo(None, "parse_error"),
     FALLBACK_CAP, "uncertain", "fallback_llm"),
    ("stylo_parse_error_low_llm", _llm(0.15), _stylo(None, "parse_error"),
     0.15, "likely_human", "fallback_llm"),
    # --- Both degraded: no confidence ---------------------------------------
    ("both_degraded", _llm(None, "parse_error"), _stylo(None, "parse_error"),
     None, "uncertain", "degraded"),
]


# Ensemble Detection stretch (Signal 3 present). Weights 0.5/0.3/0.2, renormalized
# over the usable subset; disagreement (spread > 0.40) caps the blend at 0.69.
# (name, llm_signal, stylo_signal, ppl_signal, exp_conf, exp_attr, exp_mode)
ENSEMBLE_CASES = [
    # All three usable, agree -> straight weighted blend.
    # 0.5*0.9 + 0.3*0.8 + 0.2*0.85 = 0.86
    ("ens_all_high", _llm(0.9), _stylo(0.8), _ppl(0.85),
     0.86, "likely_ai", "ensemble"),
    # 0.5*0.1 + 0.3*0.2 + 0.2*0.15 = 0.14
    ("ens_all_low", _llm(0.1), _stylo(0.2), _ppl(0.15),
     0.14, "likely_human", "ensemble"),
    # Strong disagreement (spread 0.45 > 0.40): blend 0.815 capped to 0.69.
    ("ens_conflict_capped", _llm(0.95), _stylo(0.5), _ppl(0.95),
     FALLBACK_CAP, "uncertain", "ensemble_conflict"),
    # Disagreement flagged but blend already below the cap (0.53): mode flips,
    # confidence unchanged. spread = 0.7 - 0.2 = 0.5 > 0.40.
    ("ens_conflict_below_cap", _llm(0.7), _stylo(0.2), _ppl(0.6),
     0.53, "uncertain", "ensemble_conflict"),
    # 2-of-3 usable (perplexity unavailable): renormalize over {llm, stylo}.
    # (0.5*0.9 + 0.3*0.7) / 0.8 = 0.66/0.8 = 0.825 -> 2 signals CAN reach AI band.
    ("ens_two_usable_blend", _llm(0.9), _stylo(0.7), _ppl(None, "unavailable"),
     0.825, "likely_ai", "ensemble"),
    # 2-of-3 usable AND conflicting: (0.5*0.9 + 0.3*0.1)/0.8 = 0.48/0.8 = 0.6,
    # spread 0.8 > 0.40 -> ensemble_conflict (here already under the cap).
    ("ens_two_usable_conflict", _llm(0.9), _stylo(0.1), _ppl(None, "parse_error"),
     0.6, "uncertain", "ensemble_conflict"),
    # 1-of-3 usable: single-signal fallback, capped at 0.69 (asymmetry holds).
    ("ens_one_usable_high", _llm(0.95), _stylo(None, "parse_error"),
     _ppl(None, "unavailable"), FALLBACK_CAP, "uncertain", "ensemble_fallback"),
    ("ens_one_usable_low", _llm(None, "parse_error"), _stylo(None, "parse_error"),
     _ppl(0.2), 0.2, "likely_human", "ensemble_fallback"),
    # All three degraded: no confidence.
    ("ens_all_degraded", _llm(None, "parse_error"), _stylo(None, "parse_error"),
     _ppl(None, "unavailable"), None, "uncertain", "degraded"),
]


def main():
    # Re-assert the thresholds against planning.md §3 verbatim.
    assert BAND_AI_MIN == 0.70, f"BAND_AI_MIN drifted: {BAND_AI_MIN}"
    assert BAND_HUMAN_MAX == 0.40, f"BAND_HUMAN_MAX drifted: {BAND_HUMAN_MAX}"
    assert FALLBACK_CAP == 0.69, f"FALLBACK_CAP drifted: {FALLBACK_CAP}"
    assert DISAGREE_SPREAD == 0.40, f"DISAGREE_SPREAD drifted: {DISAGREE_SPREAD}"
    print("thresholds OK: AI>=0.70, human<0.40, fallback cap 0.69, "
          "disagree spread 0.40\n")

    # Backward compatibility: passing perplexity_signal=None must reproduce the
    # two-signal path exactly (the required system is untouched by the stretch).
    for name, llm, stylo, *_ in CASES:
        a = score_confidence(llm, stylo)
        b = score_confidence(llm, stylo, None)
        assert a == b, f"legacy path diverged for {name}: {a} != {b}"
    print(f"legacy equivalence OK: perplexity=None matches 2-arg call "
          f"({len(CASES)} cases)\n")

    failures = 0

    print("--- Two-signal path (required system) ---")
    for name, llm, stylo, exp_conf, exp_attr, exp_mode in CASES:
        result = score_confidence(llm, stylo)
        failures += _check(name, result, exp_conf, exp_attr, exp_mode)

    print("\n--- Three-signal ensemble path (stretch) ---")
    for name, llm, stylo, ppl, exp_conf, exp_attr, exp_mode in ENSEMBLE_CASES:
        result = score_confidence(llm, stylo, ppl)
        failures += _check(name, result, exp_conf, exp_attr, exp_mode)

    total = len(CASES) + len(ENSEMBLE_CASES)
    print(f"\n{total - failures}/{total} cases passed.")
    sys.exit(1 if failures else 0)


def _check(name, result, exp_conf, exp_attr, exp_mode):
    """Print PASS/FAIL for one case; return 1 if it failed, else 0."""
    ok = (
        result["confidence"] == exp_conf
        and result["attribution"] == exp_attr
        and result["mode"] == exp_mode
    )
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(
        f"        got  conf={result['confidence']} "
        f"attr={result['attribution']} mode={result['mode']}"
    )
    if not ok:
        print(f"        want conf={exp_conf} attr={exp_attr} mode={exp_mode}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
