from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from promote import (
    MODEL_NAME,
    PROMOTION_TAG,
    PROMOTION_TAG_PASS_VALUE,
    SERVING_METADATA_PATH,
    SERVING_MODEL_PATH,
    TEMP_METADATA_PATH,
    TEMP_MODEL_PATH,
    clear_serving_artifact_backups,
    replace_serving_artifacts,
    restore_serving_artifact_backups,
)

TRACKING_URI = "sqlite:///mlflow.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move champion to a previously promoted registered model version.")
    parser.add_argument("--version", required=True, help="Existing registered model version that previously passed the promotion gate.")
    return parser.parse_args()


def write_serving_metadata(client: MlflowClient, model_version) -> None:
    try:
        run = client.get_run(model_version.run_id)
        run_name = str(run.data.tags.get("mlflow.runName", "unknown"))
        kind = str(run.data.params.get("kind", "unknown"))
    except Exception:
        run_name = "unknown"
        kind = "unknown"
    metadata = {
        "model_name": MODEL_NAME,
        "version": str(model_version.version),
        "run_id": str(model_version.run_id),
        "run_name": run_name,
        "kind": kind,
    }
    TEMP_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    version = str(args.version)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    try:
        model_version = client.get_model_version(MODEL_NAME, version)
    except Exception as exc:
        raise SystemExit(f"Model version {version} does not exist: {exc}") from exc

    if model_version.tags.get(PROMOTION_TAG) != PROMOTION_TAG_PASS_VALUE:
        raise SystemExit(
            f"Model version {version} has not passed the promotion gate. "
            "Rollback is allowed only to a previously promoted version."
        )

    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    except Exception as exc:
        raise SystemExit(f"Model version {version} could not be loaded: {exc}") from exc

    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_MODEL_PATH.unlink(missing_ok=True)
    TEMP_METADATA_PATH.unlink(missing_ok=True)
    joblib.dump(model, TEMP_MODEL_PATH)
    write_serving_metadata(client, model_version)

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

    print(f"ROLLBACK - champion -> version {version}")
    print(f"Serving copy: {SERVING_MODEL_PATH.as_posix()}")
    print(f"Serving metadata: {SERVING_METADATA_PATH.as_posix()}")


if __name__ == "__main__":
    main()
