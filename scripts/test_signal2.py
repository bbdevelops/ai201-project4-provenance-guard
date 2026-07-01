"""Standalone test harness for Signal 2 (planning.md §M4 verify step).

Calls analyze_stylometrics() directly (NOT through the Flask route) on the four
labeled M4 inputs plus one degenerate case, printing the combined signal score,
status, genre, and EVERY sub-metric so a miscalibrated metric can be located
before tuning (the spec's "print both signal scores separately" guidance).

Inputs 1 and 2 are verbatim from test_signal1.py so the two signals can be
compared on identical text — do they agree? Where do they diverge?

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\test_signal2.py
"""

import os
import sys

# Allow importing the package modules when run from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.stylometric_signal import analyze_stylometrics  # noqa: E402

# Each case: (text, content_type, expectation note).
CASES = {
    "clearly_ai": (
        (
            "Artificial intelligence represents a transformative paradigm shift in "
            "modern society. It is important to note that while the benefits of AI "
            "are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment."
        ),
        None,
        "expect HIGH (uniform, even sentences) -> likely_ai end",
    ),
    "clearly_human": (
        (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium in "
            "it and i was thirsty for like three hours after. my friend got the "
            "spicy version and said it was better. probably won't go back unless "
            "someone drags me there"
        ),
        None,
        "expect LOW (bursty, sparse punctuation) -> likely_human end",
    ),
    "borderline_formal_human": (
        (
            "The relationship between monetary policy and asset price inflation has "
            "been extensively studied in the literature. Central banks face a "
            "fundamental tension between their mandate for price stability and the "
            "unintended consequences of prolonged low interest rates on equity and "
            "real estate valuations."
        ),
        "academic",
        "expect HIGH-ish even with academic profile (the false-positive risk)",
    ),
    "borderline_edited_ai": (
        (
            "I've been thinking a lot about remote work lately. There are genuine "
            "tradeoffs — flexibility and no commute on one side, isolation and "
            "blurred work-life boundaries on the other. Studies show productivity "
            "varies widely by individual and role type."
        ),
        None,
        "expect MID-range",
    ),
    "degenerate_short": (
        "hi.",
        None,
        "expect status parse_error, score None",
    ),
}


def main():
    failures = 0
    for name, (text, content_type, note) in CASES.items():
        result = analyze_stylometrics(text, content_type)
        print(f"\n=== {name} ===  ({note})")
        print(f"  content_type: {content_type!r} -> genre {result['genre']!r}")
        print(f"  status:       {result['status']}")
        print(f"  score:        {result['score']}")
        metrics = result["metrics"]
        if result["status"] == "success":
            print("  raw metrics:")
            print(f"    burstiness        = {metrics['burstiness']}")
            print(f"    windowed_ttr      = {metrics['windowed_ttr']}")
            print(f"    punct_density     = {metrics['punct_density']}")
            print(f"    mean_sentence_len = {metrics['mean_sentence_len']}")
            print(f"    clause_density    = {metrics['clause_density']}")
            print(f"    ({metrics['total_words']} words, {metrics['n_sentences']} sentences)")
            print("  sub-scores (AI-ness, higher = more AI):")
            print(f"    burst_ai   = {metrics['burst_ai']}")
            print(f"    ttr_ai     = {metrics['ttr_ai']}")
            print(f"    punct_ai   = {metrics['punct_ai']}")
            print(f"    complex_ai = {metrics['complex_ai']}")
        else:
            print(f"  detail:       {metrics}")

        # Enforce Diagnostic Constraints
        if name == "clearly_ai":
            if not (result["status"] == "success" and result["score"] >= 0.70):
                print("  -> FAIL: Expected high score")
                failures += 1
        elif name == "clearly_human":
            if not (result["status"] == "success" and result["score"] < 0.40):
                print("  -> FAIL: Expected low score")
                failures += 1
        elif name == "degenerate_short":
            if result["status"] != "parse_error":
                print("  -> FAIL: Expected parse_error")
                failures += 1

    if failures:
        print(f"\n{failures} tests failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
