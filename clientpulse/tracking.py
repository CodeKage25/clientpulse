"""Mirror every registered version into MLflow.

The file registry stays the source of truth for promotion logic (so the repo
runs anywhere), and each version is also logged to MLflow as a run plus a
registered model version. Promotions move the `champion` alias, which is how
MLflow 3 replaces the old Staging/Production stages.

    mlflow ui --backend-store-uri sqlite:///mlflow.db

If MLflow is not installed, every function here is a no-op.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.pyfunc
    from mlflow import MlflowClient

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    ENABLED = os.environ.get("CLIENTPULSE_MLFLOW", "1") != "0"
except ImportError:  # pragma: no cover
    ENABLED = False

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "clientpulse-churn"
EXPERIMENT = "clientpulse-churn"


def _uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


if ENABLED:
    import warnings

    warnings.filterwarnings("ignore", message=".*Add type hints to the `predict` method")

    class ChurnPyfunc(mlflow.pyfunc.PythonModel):
        """Serves the joblib model through MLflow's standard interface."""

        def load_context(self, context):
            import joblib

            self.model = joblib.load(context.artifacts["model"])

        def predict(self, context, model_input):
            from clientpulse.features import build_features

            return self.model.predict_proba(build_features(model_input))[:, 1]


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}".replace(" ", "_").replace("+", "plus")
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def log_version(version: str, model_dir: Path, metadata: dict[str, Any]) -> str | None:
    """Log one registry version as an MLflow run and registered model version."""
    if not ENABLED:
        return None
    mlflow.set_tracking_uri(_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=version) as run:
        mlflow.set_tags(
            {
                "registry_version": version,
                "train_window": metadata.get("train_window", ""),
                "test_window": metadata.get("test_window", ""),
                "data_fingerprint": metadata.get("data_fingerprint", ""),
                "variant": metadata.get("variant", "initial"),
                "parent_version": metadata.get("parent_version", ""),
            }
        )
        mlflow.log_params({k: v for k, v in metadata.get("params", {}).items() if not isinstance(v, (dict, list))})
        mlflow.log_metrics(_flatten(metadata.get("metrics", {})))
        if "ablation_on_holdout" in metadata:
            mlflow.log_metrics(_flatten(metadata["ablation_on_holdout"], "ablation."))
        with tempfile.TemporaryDirectory() as tmp:
            slim = {k: v for k, v in metadata.items() if k != "reference_profile"}
            p = Path(tmp) / "metadata.json"
            p.write_text(json.dumps(slim, indent=2, default=str))
            mlflow.log_artifact(str(p))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ChurnPyfunc(),
            artifacts={"model": str(model_dir / "model.joblib")},
            code_paths=[str(ROOT / "clientpulse")],
            registered_model_name=MODEL_NAME,
        )
        return str(info.registered_model_version) if info.registered_model_version else run.info.run_id


def set_champion(mlflow_version: str | None, reason: str) -> None:
    if not ENABLED or mlflow_version is None:
        return
    client = MlflowClient(tracking_uri=_uri())
    client.set_registered_model_alias(MODEL_NAME, "champion", mlflow_version)
    client.set_model_version_tag(MODEL_NAME, mlflow_version, "promotion_reason", reason)
