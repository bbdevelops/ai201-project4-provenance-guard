"""Provenance Guard — Flask API.

Milestone 4 scope: the submission endpoint runs BOTH detection signals — Signal 1
(Groq LLM, semantic) and Signal 2 (stylometrics, structural) — combines them in
the isolated confidence scorer, and records both individual scores plus the
combined confidence to the structured SQLite audit log. /log surfaces it.

Deliberately NOT here yet (M5): real transparency labels, /appeal, and rate
limiting. The ``label`` field below is a clearly marked placeholder.
"""

import uuid

from flask import Flask, jsonify, request

from audit import get_log, init_db, write_entry
from scoring import score_confidence
from signals.llm_signal import classify_with_llm
from signals.stylometric_signal import analyze_stylometrics

app = Flask(__name__)
init_db()

# M4 placeholder. The real label text (three variants) is generated from the
# combined confidence score in Milestone 5.
PLACEHOLDER_LABEL = "Transparency label generated in Milestone 5."


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

    # Signal 2 — structural stylometrics (genre-aware via optional content_type).
    signal2 = analyze_stylometrics(text, content_type)
    stylo_score = signal2["score"]
    stylo_status = signal2["status"]

    # Combine both signals in the isolated scorer (blend + fallback + bands).
    verdict = score_confidence(signal1, signal2)
    attribution = verdict["attribution"]
    confidence = verdict["confidence"]

    # Audit write happens BEFORE responding so every decision is recorded.
    write_entry(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": llm_score,
            "stylo_score": stylo_score,
            "llm_status": llm_status,
            "injection_suspected": injection_suspected,
            "status": "classified",
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "signal_scores": {
                "llm_score": llm_score,
                "llm_status": llm_status,
                "stylo_score": stylo_score,
                "stylo_status": stylo_status,
            },
            "label": PLACEHOLDER_LABEL,  # placeholder until M5
        }
    )


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
