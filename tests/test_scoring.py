"""Pytest implementation for the confidence scorer (planning.md §M4 verify)."""

import pytest

from scoring import (
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

CASES = [
    ("both_high", _llm(0.9), _stylo(0.8), 0.86, "likely_ai", "blended"),
    ("both_low", _llm(0.1), _stylo(0.2), 0.14, "likely_human", "blended"),
    ("both_mid", _llm(0.5), _stylo(0.5), 0.5, "uncertain", "blended"),
    ("disagree", _llm(0.8), _stylo(0.2), 0.56, "uncertain", "blended"),
    ("llm_parse_error_high_stylo", _llm(None, "parse_error"), _stylo(0.95), FALLBACK_CAP, "uncertain", "fallback_stylo"),
    ("llm_injection_high_stylo", _llm(None, "injection_flagged"), _stylo(0.95), FALLBACK_CAP, "uncertain", "fallback_stylo"),
    ("llm_parse_error_low_stylo", _llm(None, "parse_error"), _stylo(0.2), 0.2, "likely_human", "fallback_stylo"),
    ("stylo_parse_error_high_llm", _llm(0.95), _stylo(None, "parse_error"), FALLBACK_CAP, "uncertain", "fallback_llm"),
    ("stylo_parse_error_low_llm", _llm(0.15), _stylo(None, "parse_error"), 0.15, "likely_human", "fallback_llm"),
    ("both_degraded", _llm(None, "parse_error"), _stylo(None, "parse_error"), None, "uncertain", "degraded"),
]

ENSEMBLE_CASES = [
    ("ens_all_high", _llm(0.9), _stylo(0.8), _ppl(0.85), 0.86, "likely_ai", "ensemble"),
    ("ens_all_low", _llm(0.1), _stylo(0.2), _ppl(0.15), 0.14, "likely_human", "ensemble"),
    ("ens_conflict_capped", _llm(0.95), _stylo(0.5), _ppl(0.95), FALLBACK_CAP, "uncertain", "ensemble_conflict"),
    ("ens_conflict_below_cap", _llm(0.7), _stylo(0.2), _ppl(0.6), 0.53, "uncertain", "ensemble_conflict"),
    ("ens_two_usable_blend", _llm(0.9), _stylo(0.7), _ppl(None, "unavailable"), 0.825, "likely_ai", "ensemble"),
    ("ens_two_usable_conflict", _llm(0.9), _stylo(0.1), _ppl(None, "parse_error"), 0.6, "uncertain", "ensemble_conflict"),
    ("ens_one_usable_high", _llm(0.95), _stylo(None, "parse_error"), _ppl(None, "unavailable"), FALLBACK_CAP, "uncertain", "ensemble_fallback"),
    ("ens_one_usable_low", _llm(None, "parse_error"), _stylo(None, "parse_error"), _ppl(0.2), 0.2, "likely_human", "ensemble_fallback"),
    ("ens_all_degraded", _llm(None, "parse_error"), _stylo(None, "parse_error"), _ppl(None, "unavailable"), None, "uncertain", "degraded"),
]

def test_thresholds():
    assert BAND_AI_MIN == 0.70
    assert BAND_HUMAN_MAX == 0.40
    assert FALLBACK_CAP == 0.69
    assert DISAGREE_SPREAD == 0.40

@pytest.mark.parametrize("name,llm,stylo,exp_conf,exp_attr,exp_mode", CASES)
def test_two_signal_path(name, llm, stylo, exp_conf, exp_attr, exp_mode):
    result = score_confidence(llm, stylo)
    assert result["confidence"] == exp_conf
    assert result["attribution"] == exp_attr
    assert result["mode"] == exp_mode
    
    # legacy equivalence
    result_legacy = score_confidence(llm, stylo, None)
    assert result == result_legacy

@pytest.mark.parametrize("name,llm,stylo,ppl,exp_conf,exp_attr,exp_mode", ENSEMBLE_CASES)
def test_three_signal_ensemble_path(name, llm, stylo, ppl, exp_conf, exp_attr, exp_mode):
    result = score_confidence(llm, stylo, ppl)
    assert result["confidence"] == exp_conf
    assert result["attribution"] == exp_attr
    assert result["mode"] == exp_mode
