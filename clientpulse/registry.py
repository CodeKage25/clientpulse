"""A small file-based model registry.

Same ideas as MLflow's registry (versions, stages, lineage, metrics), kept as
plain files so the repo runs anywhere. Swapping to MLflow means replacing
register/load/promote with mlflow.register_model and stage transitions.

Layout:
    models/registry.json            index of versions and which one is in production
    models/v{n}/model.joblib        the fitted model
    models/v{n}/metadata.json       lineage, metrics, params, drift reference profile
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "models"


def data_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the training data so every model can be traced to its exact inputs."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


class Registry:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    # index helpers
    def _read(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"production": None, "versions": []}

    def _write(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2))

    # public API
    def register(self, model: Any, metadata: dict[str, Any], stage: str = "candidate") -> str:
        index = self._read()
        version = f"v{len(index['versions']) + 1}"
        vdir = self.root / version
        vdir.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, vdir / "model.joblib")
        metadata = {
            **metadata,
            "version": version,
            "stage": stage,
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        index["versions"].append({"version": version, "stage": stage, "registered_at": metadata["registered_at"]})
        self._write(index)
        return version

    def promote(self, version: str, reason: str) -> None:
        """Move a version to production and archive the previous one. Keeps an audit trail."""
        index = self._read()
        previous = index["production"]
        for v in index["versions"]:
            if v["version"] == version:
                v["stage"] = "production"
                v["promotion_reason"] = reason
            elif v["version"] == previous:
                v["stage"] = "archived"
        index["production"] = version
        index.setdefault("history", []).append(
            {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "from": previous,
                "to": version,
                "reason": reason,
            }
        )
        self._write(index)
        for v in (version, previous):
            if v:
                meta_path = self.root / v / "metadata.json"
                meta = json.loads(meta_path.read_text())
                meta["stage"] = "production" if v == version else "archived"
                meta_path.write_text(json.dumps(meta, indent=2, default=str))

    def reject(self, version: str, reason: str) -> None:
        index = self._read()
        for v in index["versions"]:
            if v["version"] == version:
                v["stage"] = "rejected"
                v["rejection_reason"] = reason
        self._write(index)

    def production_version(self) -> str | None:
        return self._read()["production"]

    def load(self, version: str | None = None) -> tuple[Any, dict[str, Any]]:
        version = version or self.production_version()
        if version is None:
            raise LookupError("No model in production yet. Run the training pipeline first.")
        vdir = self.root / version
        return joblib.load(vdir / "model.joblib"), json.loads((vdir / "metadata.json").read_text())

    def index(self) -> dict[str, Any]:
        return self._read()
