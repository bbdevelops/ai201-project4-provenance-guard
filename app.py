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

from flask import Flask, jsonify, render_template_string, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit import get_dashboard_metrics, get_log, get_submission, init_db, update_status, write_entry
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


# ---------------------------------------------------------------------------
# Analytics Dashboard (stretch feature)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Provenance Guard — Analytics Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0f1117;
      color: #e4e4e7;
      min-height: 100vh;
      padding: 2rem;
    }

    h1 {
      font-size: 1.6rem;
      font-weight: 600;
      margin-bottom: .25rem;
      background: linear-gradient(135deg, #6366f1, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: #71717a;
      font-size: .85rem;
      margin-bottom: 2rem;
    }

    .live-dot {
      display: inline-block;
      width: 8px; height: 8px;
      background: #4ade80;
      border-radius: 50%;
      margin-left: .5rem;
      vertical-align: middle;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: .3; }
    }

    /* ---- Summary cards ---- */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .card {
      background: #18181b;
      border: 1px solid #27272a;
      border-radius: 12px;
      padding: 1.25rem;
      transition: border-color .2s;
    }
    .card:hover { border-color: #3f3f46; }

    .card .emoji { font-size: 1.5rem; margin-bottom: .5rem; }
    .card .value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.1;
      transition: color .3s;
    }
    .card .label {
      font-size: .75rem;
      color: #a1a1aa;
      text-transform: uppercase;
      letter-spacing: .05em;
      margin-top: .35rem;
    }
    .card .pct {
      font-size: .85rem;
      color: #71717a;
      margin-top: .15rem;
    }

    .card.ai    .value { color: #f87171; }
    .card.uncertain .value { color: #fbbf24; }
    .card.human .value { color: #4ade80; }
    .card.total .value { color: #a78bfa; }

    /* ---- Charts row ---- */
    .charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    @media (max-width: 700px) {
      .charts { grid-template-columns: 1fr; }
    }

    .chart-box {
      background: #18181b;
      border: 1px solid #27272a;
      border-radius: 12px;
      padding: 1.5rem;
    }
    .chart-box h2 {
      font-size: .95rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: #d4d4d8;
    }

    /* ---- Detail / log tables ---- */
    .detail, .audit-log {
      background: #18181b;
      border: 1px solid #27272a;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }
    .detail { max-width: 520px; }

    .detail h2, .audit-log h2 {
      font-size: .95rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: #d4d4d8;
    }
    .detail table, .audit-log table {
      width: 100%;
      border-collapse: collapse;
    }
    .detail th, .detail td,
    .audit-log th, .audit-log td {
      text-align: left;
      padding: .45rem .6rem;
      font-size: .8rem;
    }
    .detail th, .audit-log th {
      color: #a1a1aa;
      font-weight: 500;
      border-bottom: 1px solid #27272a;
      position: sticky;
      top: 0;
      background: #18181b;
    }
    .detail td, .audit-log td {
      border-bottom: 1px solid #1e1e22;
    }
    .detail tr:last-child td,
    .audit-log tr:last-child td { border-bottom: none; }

    .audit-log .scroll {
      max-height: 400px;
      overflow-y: auto;
    }

    /* Attribution badges */
    .badge {
      display: inline-block;
      padding: .15rem .5rem;
      border-radius: 6px;
      font-size: .75rem;
      font-weight: 600;
    }
    .badge.likely_ai    { background: #3b1520; color: #f87171; }
    .badge.uncertain    { background: #3b2f10; color: #fbbf24; }
    .badge.likely_human { background: #103b20; color: #4ade80; }

    .status-pill {
      display: inline-block;
      padding: .1rem .4rem;
      border-radius: 4px;
      font-size: .7rem;
    }
    .status-pill.classified    { background: #27272a; color: #a1a1aa; }
    .status-pill.under_review  { background: #3b2f10; color: #fbbf24; }

    .event-appeal { color: #818cf8; font-weight: 600; }
  </style>
</head>
<body>
  <h1>\U0001f6e1\ufe0f Provenance Guard — Analytics <span class="live-dot" title="Live — auto-refreshes every 5s"></span></h1>
  <p class="subtitle">Detection patterns, appeal rate, and security telemetry — sourced from the audit log. Updates live.</p>

  <!-- Summary cards -->
  <div class="cards">
    <div class="card total">
      <div class="emoji">\U0001f4ca</div>
      <div class="value" id="card-total">{{ total_classifications }}</div>
      <div class="label">Total Submissions</div>
    </div>
    <div class="card ai">
      <div class="emoji">\U0001f916</div>
      <div class="value" id="card-ai">{{ likely_ai_count }}</div>
      <div class="label">Likely AI</div>
      <div class="pct" id="card-ai-pct">{{ "%.1f"|format(likely_ai_count / total_classifications * 100) if total_classifications > 0 else "0.0" }}%</div>
    </div>
    <div class="card uncertain">
      <div class="emoji">\u2753</div>
      <div class="value" id="card-uncertain">{{ uncertain_count }}</div>
      <div class="label">Uncertain</div>
      <div class="pct" id="card-uncertain-pct">{{ "%.1f"|format(uncertain_count / total_classifications * 100) if total_classifications > 0 else "0.0" }}%</div>
    </div>
    <div class="card human">
      <div class="emoji">\u270d\ufe0f</div>
      <div class="value" id="card-human">{{ likely_human_count }}</div>
      <div class="label">Likely Human</div>
      <div class="pct" id="card-human-pct">{{ "%.1f"|format(likely_human_count / total_classifications * 100) if total_classifications > 0 else "0.0" }}%</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts">
    <div class="chart-box">
      <h2>Detection Pattern</h2>
      <canvas id="detectionChart"></canvas>
    </div>
    <div class="chart-box">
      <h2>Appeal Rate &amp; Injection-Flagged Rate</h2>
      <canvas id="ratesChart"></canvas>
    </div>
  </div>

  <!-- Raw numbers -->
  <div class="detail">
    <h2>Raw Metrics</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Total classifications</td><td id="raw-total">{{ total_classifications }}</td></tr>
      <tr><td>Likely AI</td><td id="raw-ai">{{ likely_ai_count }}</td></tr>
      <tr><td>Uncertain</td><td id="raw-uncertain">{{ uncertain_count }}</td></tr>
      <tr><td>Likely Human</td><td id="raw-human">{{ likely_human_count }}</td></tr>
      <tr><td>Total appeals</td><td id="raw-appeals">{{ total_appeals }}</td></tr>
      <tr><td>Appeal rate</td><td id="raw-appeal-rate">{{ "%.2f"|format(appeal_rate * 100) }}%</td></tr>
      <tr><td>Injection-flagged</td><td id="raw-injection">{{ total_injection_flagged }}</td></tr>
      <tr><td>Injection rate</td><td id="raw-injection-rate">{{ "%.2f"|format(injection_rate * 100) }}%</td></tr>
    </table>
  </div>

  <!-- Audit log -->
  <div class="audit-log">
    <h2>\U0001f4d3 Audit Log (recent entries)</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Event</th>
            <th>Content ID</th>
            <th>Attribution</th>
            <th>Confidence</th>
            <th>LLM</th>
            <th>Stylo</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="log-body">
          <!-- populated by JS on load and every poll -->
        </tbody>
      </table>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    // ---- Chart instances (created once, updated on poll) ----
    var detectionChart = new Chart(document.getElementById('detectionChart'), {
      type: 'doughnut',
      data: {
        labels: ['Likely AI', 'Uncertain', 'Likely Human'],
        datasets: [{
          data: [{{ likely_ai_count }}, {{ uncertain_count }}, {{ likely_human_count }}],
          backgroundColor: ['#f87171', '#fbbf24', '#4ade80'],
          borderColor: '#18181b',
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#a1a1aa', padding: 16, font: { size: 13 } }
          },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                var total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                var pct = total > 0 ? (ctx.raw / total * 100).toFixed(1) : '0.0';
                return ctx.label + ': ' + ctx.raw + ' (' + pct + '%)';
              }
            }
          }
        },
        cutout: '55%'
      }
    });

    var ratesChart = new Chart(document.getElementById('ratesChart'), {
      type: 'bar',
      data: {
        labels: ['Appeal Rate', 'Injection-Flagged Rate'],
        datasets: [{
          label: 'Rate (%)',
          data: [
            {{ "%.2f"|format(appeal_rate * 100) }},
            {{ "%.2f"|format(injection_rate * 100) }}
          ],
          backgroundColor: ['#818cf8', '#fb923c'],
          borderColor: ['#6366f1', '#f97316'],
          borderWidth: 1,
          borderRadius: 6,
          barPercentage: 0.5
        }]
      },
      options: {
        responsive: true,
        indexAxis: 'y',
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: { color: '#71717a', callback: v => v + '%' },
            grid: { color: '#27272a' }
          },
          y: {
            ticks: { color: '#a1a1aa', font: { size: 13 } },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return ctx.raw.toFixed(2) + '%';
              }
            }
          }
        }
      }
    });

    // ---- Helper: render a single log row ----
    function renderLogRow(e) {
      var ts = (e.timestamp || '').replace('T', ' ').replace('Z', '');
      var evt = e.event_type || 'classification';
      var evtClass = evt === 'appeal' ? ' class="event-appeal"' : '';
      var attr = e.attribution || '';
      var conf = e.confidence != null ? e.confidence.toFixed(3) : '—';
      var llm = e.llm_score != null ? e.llm_score.toFixed(2) : '—';
      var stylo = e.stylo_score != null ? e.stylo_score.toFixed(2) : '—';
      var status = e.status || '';
      var statusClass = status.replace(' ', '_');
      var cid = (e.content_id || '').substring(0, 8) + '…';
      return '<tr>'
        + '<td>' + ts + '</td>'
        + '<td' + evtClass + '>' + evt + '</td>'
        + '<td title="' + (e.content_id || '') + '">' + cid + '</td>'
        + '<td><span class="badge ' + attr + '">' + attr.replace('_', ' ') + '</span></td>'
        + '<td>' + conf + '</td>'
        + '<td>' + llm + '</td>'
        + '<td>' + stylo + '</td>'
        + '<td><span class="status-pill ' + statusClass + '">' + status.replace('_', ' ') + '</span></td>'
        + '</tr>';
    }

    // ---- Live poll: fetch metrics + log every 5 s ----
    function pct(n, total) {
      return total > 0 ? (n / total * 100).toFixed(1) : '0.0';
    }

    function refresh() {
      fetch('/dashboard/metrics')
        .then(r => r.json())
        .then(d => {
          var m = d.metrics;
          var total = m.total_classifications;

          // Cards
          document.getElementById('card-total').textContent = total;
          document.getElementById('card-ai').textContent = m.likely_ai_count;
          document.getElementById('card-ai-pct').textContent = pct(m.likely_ai_count, total) + '%';
          document.getElementById('card-uncertain').textContent = m.uncertain_count;
          document.getElementById('card-uncertain-pct').textContent = pct(m.uncertain_count, total) + '%';
          document.getElementById('card-human').textContent = m.likely_human_count;
          document.getElementById('card-human-pct').textContent = pct(m.likely_human_count, total) + '%';

          // Detection chart
          detectionChart.data.datasets[0].data = [m.likely_ai_count, m.uncertain_count, m.likely_human_count];
          detectionChart.update('none');

          // Rates chart
          var appealPct = (m.appeal_rate * 100);
          var injPct = (m.injection_rate * 100);
          ratesChart.data.datasets[0].data = [appealPct, injPct];
          ratesChart.update('none');

          // Raw metrics table
          document.getElementById('raw-total').textContent = total;
          document.getElementById('raw-ai').textContent = m.likely_ai_count;
          document.getElementById('raw-uncertain').textContent = m.uncertain_count;
          document.getElementById('raw-human').textContent = m.likely_human_count;
          document.getElementById('raw-appeals').textContent = m.total_appeals;
          document.getElementById('raw-appeal-rate').textContent = appealPct.toFixed(2) + '%';
          document.getElementById('raw-injection').textContent = m.total_injection_flagged;
          document.getElementById('raw-injection-rate').textContent = injPct.toFixed(2) + '%';

          // Audit log table
          var body = document.getElementById('log-body');
          body.innerHTML = d.log.map(renderLogRow).join('');
        })
        .catch(function() {}); // silent on network blips
    }

    // Initial load + recurring poll
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard — detection patterns, appeal rate, injection rate."""
    metrics = get_dashboard_metrics()
    return render_template_string(DASHBOARD_HTML, **metrics)


@app.route("/dashboard/metrics")
def dashboard_metrics():
    """JSON endpoint for live-polling the dashboard (metrics + recent log)."""
    return jsonify({
        "metrics": get_dashboard_metrics(),
        "log": get_log(),
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
