"""Pytest implementation for Signal 2 (planning.md §M4 verify step)."""

import pytest

from signals.stylometric_signal import analyze_stylometrics

CASES = [
    ("clearly_ai", (
        "Artificial intelligence represents a transformative paradigm shift in "
        "modern society. It is important to note that while the benefits of AI "
        "are numerous, it is equally essential to consider the ethical "
        "implications. Furthermore, stakeholders across various sectors must "
        "collaborate to ensure responsible deployment."
    ), None),
    ("clearly_human", (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in "
        "it and i was thirsty for like three hours after. my friend got the "
        "spicy version and said it was better. probably won't go back unless "
        "someone drags me there"
    ), None),
    ("borderline_formal_human", (
        "The relationship between monetary policy and asset price inflation has "
        "been extensively studied in the literature. Central banks face a "
        "fundamental tension between their mandate for price stability and the "
        "unintended consequences of prolonged low interest rates on equity and "
        "real estate valuations."
    ), "academic"),
    ("borderline_edited_ai", (
        "I've been thinking a lot about remote work lately. There are genuine "
        "tradeoffs — flexibility and no commute on one side, isolation and "
        "blurred work-life boundaries on the other. Studies show productivity "
        "varies widely by individual and role type."
    ), None),
    ("degenerate_short", (
        "hi."
    ), None),
]

@pytest.mark.parametrize("name,text,content_type", CASES)
def test_analyze_stylometrics(name, text, content_type):
    result = analyze_stylometrics(text, content_type)
    
    if name == "clearly_ai":
        assert result["status"] == "success"
        assert result["score"] >= 0.70
    elif name == "clearly_human":
        assert result["status"] == "success"
        assert result["score"] < 0.40
    elif name == "degenerate_short":
        assert result["status"] == "parse_error"
