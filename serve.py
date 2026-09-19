from __future__ import annotations

import csv
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from monitor import DRIFT_THRESHOLD, MAX_SHIFT_DISPLAY, calculate_monitoring_summary, write_html_report
from promote import THRESHOLDS

MODEL_PATH = Path("models/champion_model.joblib")
MODEL_METADATA_PATH = Path("models/champion_model_metadata.json")
PREDICTIONS_PATH = Path("monitoring/predictions.csv")
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-workshop-model"
NORMAL_REPORT_PATH = Path("monitoring/normal_report.html")
SHIFTED_REPORT_PATH = Path("monitoring/shifted_report.html")

if not MODEL_PATH.exists():
    raise RuntimeError(
        "models/champion_model.joblib is missing. Promote a passing candidate before starting the API."
    )
if not MODEL_METADATA_PATH.exists():
    raise RuntimeError(
        "models/champion_model_metadata.json is missing. Promote or roll back a passing candidate before starting the API."
    )

model = joblib.load(MODEL_PATH)
try:
    MODEL_METADATA = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise RuntimeError(f"Serving model metadata is not valid JSON: {exc}") from exc

if MODEL_METADATA.get("model_name") != MODEL_NAME or not str(MODEL_METADATA.get("version", "")).isdigit():
    raise RuntimeError("Serving model metadata does not identify a valid loaded champion version.")

MODEL_VERSION = str(MODEL_METADATA["version"])
MODEL_RUN_ID = str(MODEL_METADATA.get("run_id", ""))

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)
try:
    client.get_model_version(MODEL_NAME, MODEL_VERSION)
except Exception as exc:
    raise RuntimeError(
        f"The loaded serving model version {MODEL_VERSION} is not present in MLflow Registry."
    ) from exc

app = FastAPI(title="MLOps Fundamentals Workshop API", version="1.0")


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0, le=200)
    support_tickets: int = Field(ge=0, le=20)
    usage_score: float = Field(ge=0, le=100)
    late_payments: int = Field(ge=0, le=20)
    discount_rate: float = Field(ge=0, le=0.50)
    contract_type: Literal["monthly", "annual", "two_year"]


def append_prediction(customer: Customer, probability: float, prediction: int, latency_ms: float) -> None:
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = PREDICTIONS_PATH.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **customer.model_dump(),
        "churn_probability": round(float(probability), 6),
        "churn": int(prediction),
        "latency_ms": round(float(latency_ms), 3),
    }
    with PREDICTIONS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def alias_version(alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except Exception:
        return None


def candidate_status() -> dict[str, object] | None:
    try:
        current_candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
        run = client.get_run(current_candidate.run_id)
        metrics = {name: float(run.data.metrics[name]) for name in THRESHOLDS}
    except Exception:
        return None

    failures = [
        name
        for name, minimum in THRESHOLDS.items()
        if metrics[name] < minimum
    ]
    return {
        "version": str(current_candidate.version),
        "run_name": str(run.data.tags.get("mlflow.runName", "unknown")),
        "kind": str(run.data.params.get("kind", "unknown")),
        "metrics": metrics,
        "gate": "FAIL" if failures else "PASS",
        "failures": failures,
    }


def monitoring_status() -> dict[str, object] | None:
    try:
        summary = calculate_monitoring_summary()
    except FileNotFoundError:
        return None
    return {
        "requests": int(summary["requests"]),
        "churn_rate": float(summary["churn_rate"]),
        "average_probability": float(summary["average_probability"]),
        "p95_latency": float(summary["p95_latency"]),
        "shifts": {str(name): float(value) for name, value in summary["shifts"]},
        "alerts": list(summary["alerts"]),
        "has_alert": bool(summary["has_alert"]),
    }



def serving_copy_version() -> str | None:
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        if metadata.get("model_name") != MODEL_NAME:
            return None
        version = str(metadata.get("version", ""))
        return version if version.isdigit() else None
    except Exception:
        return None


def classify_serving_state(registry_champion: str | None, disk_version: str | None) -> tuple[str, bool]:
    if registry_champion is None:
        return "REGISTRY CHAMPION MISSING", False
    if disk_version != registry_champion:
        return "SERVING COPY OUT OF DATE", False
    if MODEL_VERSION != registry_champion:
        return "RESTART REQUIRED", False
    return "SERVING MATCH", True


def confirmed_serving_state(registry_champion: str | None) -> tuple[str, bool, str | None, str | None]:
    disk_version = serving_copy_version()
    state, matches = classify_serving_state(registry_champion, disk_version)
    if state == "SERVING COPY OUT OF DATE":
        # Avoid showing a transient state between artifact replacement and Registry alias update.
        time.sleep(0.3)
        refreshed_champion = alias_version("champion")
        refreshed_disk_version = serving_copy_version()
        state, matches = classify_serving_state(refreshed_champion, refreshed_disk_version)
        return state, matches, refreshed_disk_version, refreshed_champion
    return state, matches, disk_version, registry_champion

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "champion", "version": MODEL_VERSION}


@app.post("/predict")
def predict(customer: Customer) -> dict[str, object]:
    started = time.perf_counter()
    frame = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    latency_ms = (time.perf_counter() - started) * 1000
    append_prediction(customer, probability, prediction, latency_ms)
    return {
        "churn": bool(prediction),
        "churn_probability": round(probability, 4),
        "model": "champion",
    }


@app.get("/status-data", include_in_schema=False)
def status_data() -> dict[str, object]:
    registry_champion = alias_version("champion")
    serving_state, serving_matches, disk_version, registry_champion = confirmed_serving_state(registry_champion)
    registry_candidate = alias_version("candidate")
    return {
        "api_loaded_version": MODEL_VERSION,
        "serving_copy_version": disk_version,
        "registry_champion": registry_champion,
        "registry_candidate": registry_candidate,
        "serving_state": serving_state,
        "serving_matches_registry": serving_matches,
        "candidate": candidate_status(),
        "monitoring": monitoring_status(),
        "reports": {
            "normal": NORMAL_REPORT_PATH.exists(),
            "shifted": SHIFTED_REPORT_PATH.exists(),
        },
    }

@app.post("/monitoring/reset", include_in_schema=False)
def reset_monitoring() -> dict[str, str]:
    PREDICTIONS_PATH.unlink(missing_ok=True)
    return {"status": "reset"}


@app.post("/monitoring/report/{name}", include_in_schema=False)
def save_monitoring_report(name: Literal["normal", "shifted"]) -> dict[str, str]:
    try:
        summary = calculate_monitoring_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = NORMAL_REPORT_PATH if name == "normal" else SHIFTED_REPORT_PATH
    write_html_report(path, summary)
    return {"status": "saved", "report": f"/reports/{name}"}


@app.get("/reports/{name}", include_in_schema=False)
def report(name: Literal["normal", "shifted"]):
    path = NORMAL_REPORT_PATH if name == "normal" else SHIFTED_REPORT_PATH
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} monitoring report has not been generated yet")
    return FileResponse(path, media_type="text/html")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLOps Workshop Dashboard</title>
<style>
:root { --bg:#f4f7f9; --card:#fff; --border:#dbe3e8; --ink:#17212b; --muted:#667481; --ok:#176b3a; --okbg:#e8f7ee; --bad:#a72828; --badbg:#fdeaea; --accent:#1d6f8a; }
* { box-sizing:border-box; }
body { margin:0; font-family:Arial,sans-serif; background:var(--bg); color:var(--ink); }
main { max-width:1120px; margin:28px auto; padding:0 20px 40px; }
header { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; margin-bottom:20px; }
h1 { margin:0 0 6px; }
.sub { color:var(--muted); margin:0; }
.actions { display:flex; flex-wrap:wrap; gap:8px; }
a, button { border:1px solid var(--border); background:white; color:var(--ink); border-radius:8px; padding:9px 12px; text-decoration:none; cursor:pointer; font-weight:600; }
a:hover,button:hover { border-color:#9fb1bc; }
.grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; }
.card h2 { font-size:16px; margin:0 0 14px; }
.big { font-size:30px; font-weight:700; }
.label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.row { display:flex; justify-content:space-between; gap:12px; margin:8px 0; }
.badge { display:inline-block; padding:5px 9px; border-radius:999px; font-weight:700; font-size:12px; }
.ok { color:var(--ok); background:var(--okbg); }
.bad { color:var(--bad); background:var(--badbg); }
.neutral { color:#4d5c66; background:#eef2f4; }
.wide { grid-column:span 2; }
.monitor { grid-column:1 / -1; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:12px; }
.metric { border:1px solid var(--border); border-radius:8px; padding:12px; }
.metric strong { display:block; font-size:20px; margin-top:4px; }
.feature { margin:12px 0; }
.feature-head { display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; }
.track { height:12px; background:#e8edf1; border-radius:6px; position:relative; overflow:hidden; }
.bar { height:100%; background:#4c9f70; }
.bar.badbar { background:#d9534f; }
.threshold { position:absolute; top:0; bottom:0; width:2px; background:#202a34; z-index:2; }
#prediction-result { white-space:pre-wrap; background:#f7f9fa; border-radius:8px; padding:10px; min-height:44px; margin-top:10px; font-family:Consolas,monospace; font-size:13px; }
footer { color:var(--muted); margin-top:14px; font-size:12px; }
@media(max-width:850px){ .grid{grid-template-columns:1fr;} .wide,.monitor{grid-column:auto;} .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));} header{display:block;} .actions{margin-top:14px;} }
</style>
</head>
<body>
<main>
<header>
  <div><h1>MLOps Workshop Dashboard</h1><p class="sub">A live view of serving state, registry aliases, candidate promotion status, and monitoring.</p></div>
  <div class="actions">
    <a href="/docs" target="_blank">Open Swagger</a>
    <a href="http://127.0.0.1:5000" target="_blank">Open MLflow</a>
    <button onclick="loadStatus()">Refresh now</button>
  </div>
</header>
<section class="grid">
  <div class="card"><div class="label">Live API model</div><div class="big" id="api-version">-</div><div id="serving-state" class="badge neutral">loading</div></div>
  <div class="card"><div class="label">Registry champion</div><div class="big" id="champion-version">-</div><p class="sub">Movable production alias</p></div>
  <div class="card"><div class="label">Registry candidate</div><div class="big" id="candidate-version">-</div><p class="sub">Latest evaluation target</p></div>

  <div class="card wide">
    <h2>Current candidate promotion gate</h2>
    <div class="row"><span>Run</span><strong id="candidate-run">-</strong></div>
    <div class="row"><span>Model kind</span><strong id="candidate-kind">-</strong></div>
    <div class="row"><span>Accuracy</span><strong id="m-accuracy">-</strong></div>
    <div class="row"><span>Recall</span><strong id="m-recall">-</strong></div>
    <div class="row"><span>ROC AUC</span><strong id="m-roc_auc">-</strong></div>
    <div id="gate-state" class="badge neutral">no candidate evidence</div>
  </div>

  <div class="card">
    <h2>Try the live model</h2>
    <p class="sub">Send one demo customer through the same /predict endpoint used by Swagger.</p>
    <button onclick="sendDemo()">Send demo prediction</button>
    <div id="prediction-result">No request sent yet.</div>
  </div>

  <div class="card monitor">
    <div class="row"><h2 style="margin:0">Current monitoring sample</h2><div class="actions"><button onclick="resetMonitoring()">Reset sample</button><button onclick="saveReport('normal')">Save as normal</button><button onclick="saveReport('shifted')">Save as shifted</button><a id="normal-link" href="/reports/normal" target="_blank">Open normal</a><a id="shifted-link" href="/reports/shifted" target="_blank">Open shifted</a></div></div>
    <div id="monitor-empty" class="sub">No prediction traffic logged yet.</div>
    <div id="monitor-content" style="display:none">
      <div id="monitor-state" class="badge neutral">-</div>
      <div class="metric-grid">
        <div class="metric"><span class="label">Requests</span><strong id="reqs">-</strong></div>
        <div class="metric"><span class="label">Churn rate</span><strong id="churn-rate">-</strong></div>
        <div class="metric"><span class="label">Avg probability</span><strong id="avg-prob">-</strong></div>
        <div class="metric"><span class="label">p95 latency</span><strong id="p95">-</strong></div>
      </div>
      <div id="features"></div>
    </div>
  </div>
</section>
<footer>Auto-refresh: every 2 seconds. SERVING COPY OUT OF DATE means the on-disk serving artifact no longer matches the Registry champion. RESTART REQUIRED means the artifact is current, but the running API must be restarted.</footer>
</main>
<script>
const THRESHOLD = 0.75;
const MAX_SHIFT = 2.5;
function badge(el, text, ok){ el.textContent=text; el.className='badge '+(ok===true?'ok':ok===false?'bad':'neutral'); }
function fmt(v, n=3){ return Number(v).toFixed(n); }
async function loadStatus(){
  try{
    const r=await fetch('/status-data',{cache:'no-store'}); const s=await r.json();
    document.getElementById('api-version').textContent='v'+s.api_loaded_version;
    document.getElementById('champion-version').textContent=s.registry_champion ? 'v'+s.registry_champion : '-';
    document.getElementById('candidate-version').textContent=s.registry_candidate ? 'v'+s.registry_candidate : '-';
    badge(document.getElementById('serving-state'), s.serving_state, s.serving_matches_registry);
    if(s.candidate){
      document.getElementById('candidate-run').textContent=s.candidate.run_name;
      document.getElementById('candidate-kind').textContent=s.candidate.kind;
      document.getElementById('m-accuracy').textContent=fmt(s.candidate.metrics.accuracy)+' / 0.82';
      document.getElementById('m-recall').textContent=fmt(s.candidate.metrics.recall)+' / 0.70';
      document.getElementById('m-roc_auc').textContent=fmt(s.candidate.metrics.roc_auc)+' / 0.85';
      badge(document.getElementById('gate-state'), 'PROMOTION '+s.candidate.gate, s.candidate.gate==='PASS');
    }
    if(s.monitoring){
      document.getElementById('monitor-empty').style.display='none'; document.getElementById('monitor-content').style.display='block';
      badge(document.getElementById('monitor-state'), s.monitoring.has_alert?'ALERT: POSSIBLE DRIFT':'OK: NO DRIFT ALERT', !s.monitoring.has_alert);
      document.getElementById('reqs').textContent=s.monitoring.requests;
      document.getElementById('churn-rate').textContent=fmt(s.monitoring.churn_rate);
      document.getElementById('avg-prob').textContent=fmt(s.monitoring.average_probability);
      document.getElementById('p95').textContent=Number(s.monitoring.p95_latency).toFixed(2)+' ms';
      const f=document.getElementById('features'); f.innerHTML='';
      for(const [name,shift] of Object.entries(s.monitoring.shifts)){
        const pct=Math.min(Number(shift)/MAX_SHIFT*100,100); const th=Math.min(THRESHOLD/MAX_SHIFT*100,100); const bad=Number(shift)>=THRESHOLD;
        f.insertAdjacentHTML('beforeend',`<div class="feature"><div class="feature-head"><span>${name}</span><strong>${Number(shift).toFixed(2)}σ</strong></div><div class="track"><div class="threshold" style="left:${th}%"></div><div class="bar ${bad?'badbar':''}" style="width:${pct}%"></div></div></div>`);
      }
    } else { document.getElementById('monitor-empty').style.display='block'; document.getElementById('monitor-content').style.display='none'; }
    document.getElementById('normal-link').style.opacity=s.reports.normal?'1':'.35';
    document.getElementById('shifted-link').style.opacity=s.reports.shifted?'1':'.35';
  } catch(e){ badge(document.getElementById('serving-state'),'dashboard error',false); }
}

async function resetMonitoring(){
  await fetch('/monitoring/reset',{method:'POST'}); document.getElementById('prediction-result').textContent='Monitoring sample reset.'; await loadStatus();
}
async function saveReport(name){
  const r=await fetch('/monitoring/report/'+name,{method:'POST'});
  const data=await r.json();
  if(!r.ok){ document.getElementById('prediction-result').textContent='Could not save report: '+(data.detail||r.status); return; }
  document.getElementById('prediction-result').textContent='Saved '+name+' report. Use the Open '+name+' button.'; await loadStatus();
}
async function sendDemo(){
  const payload={tenure_months:12,monthly_spend:95,support_tickets:3,usage_score:55,late_payments:1,discount_rate:0.10,contract_type:'monthly'};
  const box=document.getElementById('prediction-result'); box.textContent='Sending...';
  try{ const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const data=await r.json(); box.textContent=JSON.stringify(data,null,2); await loadStatus(); }
  catch(e){ box.textContent='Request failed: '+e; }
}
loadStatus(); setInterval(loadStatus,2000);
</script>
</body>
</html>"""
    )
