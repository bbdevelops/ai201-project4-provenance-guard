"""Provenance Guard — Flask API.

Milestone 3 scope: the submission endpoint wired to Signal 1 (Groq LLM), a
structured SQLite audit log, and a /log endpoint to surface it.

Deliberately NOT here yet (M4/M5): Signal 2 / stylometrics, the real confidence
scorer, real transparency labels, /appeal, and rate limiting. The ``confidence``
and ``label`` fields below are clearly marked placeholders.
"""

import uuid

from flask import Flask, jsonify, request

from audit import get_log, init_db, write_entry
from signals.llm_signal import classify_with_llm

app = Flask(__name__)
init_db()

# M3 placeholder. The real label text (three variants) is generated from the
# combined confidence score in Milestone 5.
PLACEHOLDER_LABEL = "Transparency label generated in Milestone 5."


def _placeholder_attribution(llm_score):
    """Temporary stand-in for the M4 combined scorer.

    Applies the planning.md §3 three bands to the Signal-1 score ALONE so the
    response carries a meaningful attribution field now. M4 replaces this with
    the real combined-signal scorer; do not treat this as final scoring.
    """
    if llm_score is None:
        return "uncertain"
    if llm_score >= 0.70:
        return "likely_ai"
    if llm_score < 0.40:
        return "likely_human"
    return "uncertain"


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    creator_id = data.get("creator_id")
    content_type = data.get("content_type")  # optional genre hint (used in M4)

    # Validate required fields.
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    content_id = str(uuid.uuid4())

    # Signal 1 — semantic LLM classification (returns standardized contract).
    signal1 = classify_with_llm(text)
    llm_score = signal1["score"]
    llm_status = signal1["status"]
    injection_suspected = 1 if signal1.get("marker") else 0

    # Placeholder attribution/confidence/label (see notes above; real in M4/M5).
    attribution = _placeholder_attribution(llm_score)
    confidence = llm_score  # placeholder: Signal-1 score stands in for now

    # Audit write happens BEFORE responding so every decision is recorded.
    write_entry(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": llm_score,
            "stylo_score": None,  # Signal 2 arrives in M4
            "llm_status": llm_status,
            "injection_suspected": injection_suspected,
            "status": "classified",
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,  # placeholder until M4
            "signal_scores": {"llm_score": llm_score, "llm_status": llm_status},
            "label": PLACEHOLDER_LABEL,  # placeholder until M5
        }
    )


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
