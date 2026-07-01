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

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit import get_dashboard_metrics, get_log, get_submission, init_db, update_status, write_entry, get_pending_appeals, resolve_appeal, get_appeal_details
from labels import generate_label
from scoring import score_confidence
from signals.llm_signal import classify_with_llm
from signals.perplexity_signal import analyze_perplexity
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

    # Signal 3 — GPT-2 perplexity (Ensemble Detection stretch). Returns "disabled"
    # fast (no torch import) when ENABLE_PERPLEXITY_SIGNAL is off; in that case we
    # pass None to the scorer so the required two-signal path runs unchanged.
    signal3 = analyze_perplexity(text)
    ppl_score = signal3["score"]
    ppl_status = signal3["status"]
    sig3 = signal3 if ppl_status != "disabled" else None

    # Combine signals in the isolated scorer (blend + fallback + bands). With
    # Signal 3 enabled this is the ensemble path; otherwise the two-signal path.
    verdict = score_confidence(signal1, signal2, sig3)
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
            "perplexity_score": ppl_score,
            "perplexity_status": ppl_status,
            "llm_status": llm_status,
            "injection_suspected": injection_suspected,
            "status": "classified",
            "text": text,
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
                "perplexity_score": ppl_score,
                "perplexity_status": ppl_status,
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


@app.route("/api/appeals/pending", methods=["GET"])
def pending_appeals():
    """Returns a list of all appeals currently 'under_review'."""
    return jsonify({"appeals": get_pending_appeals()})


@app.route("/api/appeals/resolve", methods=["POST"])
def api_resolve_appeal():
    """Resolves an appeal with a new attribution and reviewer note."""
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    new_attribution = data.get("new_attribution")
    reviewer_note = data.get("reviewer_note")

    if not content_id or not new_attribution or not reviewer_note:
        return jsonify({"error": "Fields 'content_id', 'new_attribution', and 'reviewer_note' are required."}), 400

    if new_attribution not in ["likely_human", "uncertain", "likely_ai"]:
        return jsonify({"error": "Invalid attribution."}), 400

    success = resolve_appeal(content_id, new_attribution, reviewer_note)
    if not success:
        return jsonify({"error": "Appeal not found or already resolved."}), 404

    return jsonify({"success": True})


@app.route("/api/appeal/<content_id>", methods=["GET"])
def appeal_status(content_id):
    """Returns the current status, attribution, and reviewer_note of a submission."""
    # We can get the most recent log entry for this content_id to see its status
    submission = get_submission(content_id)
    if not submission:
        return jsonify({"error": "Unknown content_id."}), 404
        
    import sqlite3
    from config import DB_PATH
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get the latest entry to determine current status
        row = conn.execute(
            "SELECT status, attribution, reviewer_note FROM audit_log WHERE content_id = ? ORDER BY id DESC LIMIT 1",
            (content_id,)
        ).fetchone()
        
    if not row:
        return jsonify({"error": "No records found."}), 404
        
    return jsonify({
        "status": row["status"],
        "attribution": row["attribution"],
        "reviewer_note": row["reviewer_note"]
    })


# ---------------------------------------------------------------------------
# Analytics Dashboard & Dummy UI
# ---------------------------------------------------------------------------

@app.route("/dashboard")
def dashboard():
    """Analytics dashboard — detection patterns, appeal rate, injection rate."""
    metrics = get_dashboard_metrics()
    return render_template("dashboard.html", **metrics)


@app.route("/review/<content_id>")
def review_appeal(content_id):
    """Dedicated page for a site manager to review an appeal."""
    appeal = get_appeal_details(content_id)
    if not appeal:
        return "Appeal not found", 404
    return render_template("review.html", appeal=appeal)


@app.route("/dashboard/metrics")
def dashboard_metrics():
    """JSON endpoint for live-polling the dashboard (metrics + recent log)."""
    return jsonify({
        "metrics": get_dashboard_metrics(),
        "log": get_log(),
    })


@app.route("/dummy")
def dummy():
    """Dummy creative writing platform UI."""
    return render_template("dummy.html")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
