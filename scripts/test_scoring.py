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
    FALLBACK_CAP,
    score_confidence,
)


def _llm(score, status="success"):
    return {"score": score, "status": status, "rationale": None, "marker": None}


def _stylo(score, status="success"):
    return {"score": score, "status": status, "metrics": {}, "genre": "prose"}


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


def main():
    # Re-assert the thresholds against planning.md §3 verbatim.
    assert BAND_AI_MIN == 0.70, f"BAND_AI_MIN drifted: {BAND_AI_MIN}"
    assert BAND_HUMAN_MAX == 0.40, f"BAND_HUMAN_MAX drifted: {BAND_HUMAN_MAX}"
    assert FALLBACK_CAP == 0.69, f"FALLBACK_CAP drifted: {FALLBACK_CAP}"
    print("thresholds OK: AI>=0.70, human<0.40, fallback cap 0.69\n")

    failures = 0
    for name, llm, stylo, exp_conf, exp_attr, exp_mode in CASES:
        result = score_confidence(llm, stylo)
        ok = (
            result["confidence"] == exp_conf
            and result["attribution"] == exp_attr
            and result["mode"] == exp_mode
        )
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name}")
        print(
            f"        got  conf={result['confidence']} "
            f"attr={result['attribution']} mode={result['mode']}"
        )
        if not ok:
            print(
                f"        want conf={exp_conf} attr={exp_attr} mode={exp_mode}"
            )

    print(f"\n{len(CASES) - failures}/{len(CASES)} cases passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
