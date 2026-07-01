"""Pytest implementation for Signal 3 — GPT-2 perplexity (Ensemble stretch)."""

import pytest
import config
from signals.perplexity_signal import analyze_perplexity

CASES = [
    ("clearly_ai", (
        "Artificial intelligence represents a transformative paradigm shift in "
        "modern society. It is important to note that while the benefits of AI "
        "are numerous, it is equally essential to consider the ethical "
        "implications. Furthermore, stakeholders across various sectors must "
        "collaborate to ensure responsible deployment."
    )),
    ("clearly_human", (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in "
        "it and i was thirsty for like three hours after. my friend got the "
        "spicy version and said it was better. probably won't go back unless "
        "someone drags me there"
    )),
    ("borderline_formal_human", (
        "The relationship between monetary policy and asset price inflation has "
        "been extensively studied in the literature. Central banks face a "
        "fundamental tension between their mandate for price stability and the "
        "unintended consequences of prolonged low interest rates on equity and "
        "real estate valuations."
    )),
    ("borderline_edited_ai", (
        "I've been thinking a lot about remote work lately. There are genuine "
        "tradeoffs — flexibility and no commute on one side, isolation and "
        "blurred work-life boundaries on the other. Studies show productivity "
        "varies widely by individual and role type."
    )),
    ("degenerate_short", (
        "hi there friend."
    )),
]

@pytest.mark.parametrize("name,text", CASES)
def test_analyze_perplexity(name, text):
    if not config.ENABLE_PERPLEXITY_SIGNAL:
        pytest.skip("Signal 3 is disabled. Set ENABLE_PERPLEXITY_SIGNAL=true to run.")

    result = analyze_perplexity(text)
    
    if name == "clearly_ai":
        assert result["status"] == "success"
        assert result["score"] >= 0.70
    elif name == "clearly_human":
        assert result["status"] == "success"
        assert result["score"] < 0.40
    elif name == "degenerate_short":
        assert result["status"] == "parse_error"
