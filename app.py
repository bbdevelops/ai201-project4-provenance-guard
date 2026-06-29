"""Provenance Guard — Flask API.

The submission endpoint runs BOTH detection signals — Signal 1 (Groq LLM,
semantic) and Signal 2 (stylometrics, structural) — combines them in the isolated
confidence scorer, maps the result to a plain-language transparency label, and
records every decision to the structured SQLite audit log. /log surfaces it.

Milestone 5 production layer:
  - real transparency labels (three confidence-driven variants, see labels.py),
  - the /appeal endpoint (creators contest a classification),
  - IP-based rate limiting via Flask-Limiter on both POST endpoints.
"""

import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit import get_log, get_submission, init_db, update_status, write_entry
from labels import generate_label
from scoring import score_confidence
from signals.llm_signal import classify_with_llm
from signals.stylometric_signal import analyze_stylometrics

app = Flask(__name__)
init_db()

# IP-based rate limiting (Flask-Limiter). In-memory storage is fine for local
# dev / grading; a production deploy would point storage_uri at Redis. The
# per-creator_id interval check from planning.md §1/§6 is documented future work.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# A real writer submits their own work infrequently; 10/minute absorbs normal
# editing bursts while 100/day blocks a script flooding the system. Applied to
# both POST endpoints.
SUBMIT_LIMITS = "10 per minute;100 per day"


@app.route("/submit", methods=["POST"])
@limiter.limit(SUBMIT_LIMITS)
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
            "label": generate_label(confidence),
        }
    )


@app.route("/appeal", methods=["POST"])
@limiter.limit(SUBMIT_LIMITS)
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")

    # Validate required fields.
    if not content_id:
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        return (
            jsonify({"error": "Field 'creator_reasoning' is required and must be non-empty."}),
            400,
        )

    # Look up the original classification; reject unknown content_ids.
    original = get_submission(content_id)
    if original is None:
        return jsonify({"error": f"Unknown content_id: {content_id}"}), 404

    # Flip the original decision's status, then log the appeal BESIDE it —
    # carrying the original scores so a reviewer sees full context. Both happen
    # before responding, so the record exists even if the client disconnects.
    update_status(content_id, "under_review")
    write_entry(
        {
            "content_id": content_id,
            "creator_id": original.get("creator_id"),
            "event_type": "appeal",
            "attribution": original.get("attribution"),
            "confidence": original.get("confidence"),
            "llm_score": original.get("llm_score"),
            "stylo_score": original.get("stylo_score"),
            "llm_status": original.get("llm_status"),
            "status": "under_review",
            "appeal_reasoning": creator_reasoning,
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received. This content is now under review.",
        }
    )


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
