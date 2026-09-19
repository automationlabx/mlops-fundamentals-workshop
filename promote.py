from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-workshop-model"
CANDIDATE_PATH = Path("candidate.json")
SERVING_MODEL_PATH = Path("models/champion_model.joblib")
SERVING_METADATA_PATH = Path("models/champion_model_metadata.json")
TEMP_MODEL_PATH = Path("models/.champion_model.joblib.tmp")
TEMP_METADATA_PATH = Path("models/.champion_model_metadata.json.tmp")
BACKUP_MODEL_PATH = Path("models/.champion_model.joblib.bak")
BACKUP_METADATA_PATH = Path("models/.champion_model_metadata.json.bak")
PROMOTION_TAG = "promotion_gate"
PROMOTION_TAG_PASS_VALUE = "passed"
PROMOTION_TAG_FAIL_VALUE = "failed"

THRESHOLDS = {
    "accuracy": 0.82,
    "recall": 0.70,
    "roc_auc": 0.85,
}


def read_candidate() -> dict[str, object]:
    if not CANDIDATE_PATH.exists():
        raise SystemExit("candidate.json is missing. Train a model with --candidate first.")
    try:
        candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"candidate.json is not valid JSON: {exc}") from exc

    required = {"model_name", "version", "run_id", "run_name", "kind", "data_sha256"}
    missing = sorted(required - set(candidate))
    if missing:
        raise SystemExit(f"candidate.json is missing required fields: {', '.join(missing)}")
    if candidate["model_name"] != MODEL_NAME:
        raise SystemExit(f"candidate.json points to {candidate['model_name']!r}, expected {MODEL_NAME!r}.")
    return candidate


def get_run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise SystemExit(f"Candidate MLflow run {run_id} could not be loaded: {exc}") from exc
    metrics = run.data.metrics
    missing = sorted(set(THRESHOLDS) - set(metrics))
    if missing:
        raise SystemExit(f"Candidate MLflow run is missing required metrics: {', '.join(missing)}")
    return {name: float(metrics[name]) for name in THRESHOLDS}


def write_serving_metadata(version: str, run_id: str, run_name: str, kind: str) -> None:
    metadata = {
        "model_name": MODEL_NAME,
        "version": str(version),
        "run_id": str(run_id),
        "run_name": str(run_name),
        "kind": str(kind),
    }
    TEMP_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def safe_replace(src: Path, dst: Path, *, attempts: int = 8, delay_seconds: float = 0.05) -> None:
    """Replace a file with short retries for Windows file-lock timing races."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error


def backup_existing_serving_artifacts() -> None:
    for backup in (BACKUP_MODEL_PATH, BACKUP_METADATA_PATH):
        backup.unlink(missing_ok=True)
    if SERVING_MODEL_PATH.exists():
        shutil.copy2(SERVING_MODEL_PATH, BACKUP_MODEL_PATH)
    if SERVING_METADATA_PATH.exists():
        shutil.copy2(SERVING_METADATA_PATH, BACKUP_METADATA_PATH)


def restore_serving_artifact_backups() -> None:
    if BACKUP_MODEL_PATH.exists():
        safe_replace(BACKUP_MODEL_PATH, SERVING_MODEL_PATH)
    else:
        SERVING_MODEL_PATH.unlink(missing_ok=True)
    if BACKUP_METADATA_PATH.exists():
        safe_replace(BACKUP_METADATA_PATH, SERVING_METADATA_PATH)
    else:
        SERVING_METADATA_PATH.unlink(missing_ok=True)


def clear_serving_artifact_backups() -> None:
    BACKUP_MODEL_PATH.unlink(missing_ok=True)
    BACKUP_METADATA_PATH.unlink(missing_ok=True)


def replace_serving_artifacts() -> None:
    backup_existing_serving_artifacts()
    try:
        safe_replace(TEMP_MODEL_PATH, SERVING_MODEL_PATH)
        safe_replace(TEMP_METADATA_PATH, SERVING_METADATA_PATH)
    except Exception:
        restore_serving_artifact_backups()
        TEMP_MODEL_PATH.unlink(missing_ok=True)
        TEMP_METADATA_PATH.unlink(missing_ok=True)
        raise


def main() -> None:
    candidate = read_candidate()
    version = str(candidate["version"])
    run_id = str(candidate["run_id"])

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    try:
        current_candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    except Exception as exc:
        raise SystemExit(f"Candidate alias is missing in MLflow Registry: {exc}") from exc

    if str(current_candidate.version) != version:
        raise SystemExit(
            "candidate.json does not match the current candidate alias. "
            "Run the intended training command again."
        )
    if str(current_candidate.run_id) != run_id:
        raise SystemExit(
            "candidate.json does not match the MLflow run behind the current candidate version. "
            "Run the intended training command again."
        )

    # candidate.json is only a pointer. Authoritative metrics come from MLflow.
    metrics = get_run_metrics(client, run_id)
    failures = [
        f"{name}={metrics[name]:.3f} < {minimum:.2f}"
        for name, minimum in THRESHOLDS.items()
        if metrics[name] < minimum
    ]

    if failures:
        client.set_model_version_tag(MODEL_NAME, version, PROMOTION_TAG, PROMOTION_TAG_FAIL_VALUE)
        print(f"FAIL - candidate version {version} was not promoted")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    # Validate and stage the serving copy before changing the champion alias.
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    except Exception as exc:
        raise SystemExit(f"Candidate model version {version} could not be loaded: {exc}") from exc

    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_MODEL_PATH.unlink(missing_ok=True)
    TEMP_METADATA_PATH.unlink(missing_ok=True)
    joblib.dump(model, TEMP_MODEL_PATH)
    write_serving_metadata(version, run_id, str(candidate["run_name"]), str(candidate["kind"]))

    client.set_model_version_tag(MODEL_NAME, version, PROMOTION_TAG, PROMOTION_TAG_PASS_VALUE)
    replace_serving_artifacts()

    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    except Exception:
        restore_serving_artifact_backups()
        raise
    finally:
        clear_serving_artifact_backups()
        TEMP_MODEL_PATH.unlink(missing_ok=True)
        TEMP_METADATA_PATH.unlink(missing_ok=True)

    print(f"PASS - version {version} promoted to champion")
    print(f"Serving copy: {SERVING_MODEL_PATH.as_posix()}")
    print(f"Serving metadata: {SERVING_METADATA_PATH.as_posix()}")


if __name__ == "__main__":
    main()
