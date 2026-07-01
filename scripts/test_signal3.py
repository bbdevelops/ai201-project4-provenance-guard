"""Standalone test harness for Signal 3 — GPT-2 perplexity (Ensemble stretch).

Calls analyze_perplexity() directly (NOT through the Flask route) on the four
labeled M4 inputs plus one degenerate case, printing the perplexity, the mapped
AI-likelihood score, status, and token count. Reuses the same texts as
test_signal2.py so all three signals can be compared on identical inputs.

This script never hard-fails: if ENABLE_PERPLEXITY_SIGNAL is off it prints
"disabled"; if torch/transformers/the model can't load it prints "unavailable".
That is the point of the opt-in design — the required system runs without it.

To exercise the real model:
    pip install -r requirements-ensemble.txt
    set ENABLE_PERPLEXITY_SIGNAL=true   (PowerShell: $env:ENABLE_PERPLEXITY_SIGNAL="true")
    .venv\\Scripts\\python.exe scripts\\test_signal3.py

Expectation: clearly-AI text is more predictable (LOWER perplexity -> HIGHER
score) than clearly-human text. GPT-2 Small (~500 MB) downloads on first use.
"""

import os
import sys

# Allow importing the package modules when run from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from signals.perplexity_signal import analyze_perplexity  # noqa: E402

# Same texts as test_signal2.py: (text, expectation note).
CASES = {
    "clearly_ai": (
        (
            "Artificial intelligence represents a transformative paradigm shift in "
            "modern society. It is important to note that while the benefits of AI "
            "are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment."
        ),
        "expect LOW perplexity -> HIGH score (predictable, AI-like)",
    ),
    "clearly_human": (
        (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium in "
            "it and i was thirsty for like three hours after. my friend got the "
            "spicy version and said it was better. probably won't go back unless "
            "someone drags me there"
        ),
        "expect HIGH perplexity -> LOW score (surprising, human-like)",
    ),
    "borderline_formal_human": (
        (
            "The relationship between monetary policy and asset price inflation has "
            "been extensively studied in the literature. Central banks face a "
            "fundamental tension between their mandate for price stability and the "
            "unintended consequences of prolonged low interest rates on equity and "
            "real estate valuations."
        ),
        "expect MID/LOW perplexity (formal but human) — the false-positive risk",
    ),
    "borderline_edited_ai": (
        (
            "I've been thinking a lot about remote work lately. There are genuine "
            "tradeoffs — flexibility and no commute on one side, isolation and "
            "blurred work-life boundaries on the other. Studies show productivity "
            "varies widely by individual and role type."
        ),
        "expect MID-range",
    ),
    "degenerate_short": (
        "hi there friend.",
        "expect status parse_error (below the ~20-word floor), score None",
    ),
}


def main():
    print(f"ENABLE_PERPLEXITY_SIGNAL = {config.ENABLE_PERPLEXITY_SIGNAL}")
    print(f"PERPLEXITY_MODEL         = {config.PERPLEXITY_MODEL!r}\n")
    if not config.ENABLE_PERPLEXITY_SIGNAL:
        print("Signal 3 is disabled — every case returns status 'disabled'.")
        print("Set ENABLE_PERPLEXITY_SIGNAL=true and install "
              "requirements-ensemble.txt to run the real model.\n")

    failures = 0
    for name, (text, note) in CASES.items():
        result = analyze_perplexity(text)
        print(f"=== {name} ===  ({note})")
        print(f"  status: {result['status']}")
        print(f"  score:  {result['score']}")
        metrics = result["metrics"]
        if result["status"] == "success":
            print(f"  perplexity          = {metrics['perplexity']}")
            print(f"  mean_log_likelihood = {metrics['mean_log_likelihood']}")
            print(f"  n_tokens            = {metrics['n_tokens']}")
        elif metrics:
            print(f"  detail: {metrics}")
        print()

        if config.ENABLE_PERPLEXITY_SIGNAL:
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
