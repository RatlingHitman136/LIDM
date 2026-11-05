"""Thin MLflow wrapper so experiment code does not scatter `mlflow.*` calls.

Local sqlite backend (override with MLFLOW_TRACKING_URI). View with scripts/mlflow_ui.sh.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def env_tags() -> dict[str, str]:
    import torch
    import transformers

    tags = {
        "git_commit": _git("rev-parse", "--short", "HEAD") or "no-git",
        "git_dirty": str(bool(_git("status", "--porcelain"))) if _git("rev-parse", "HEAD") else "n/a",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import laya

        tags["laya"] = laya.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        tags["gpu"] = torch.cuda.get_device_name(0)
    return tags


class Tracker:
    """Context manager around an MLflow run. `enabled=False` makes every call a no-op."""

    def __init__(self, experiment: str, run_name: str, tags: dict[str, str] | None = None,
                 enabled: bool = True, parent: "Tracker | None" = None):
        self.enabled = enabled
        self.experiment, self.run_name = experiment, run_name
        self.tags = tags or {}
        self.parent = parent
        self.run = None

    def __enter__(self) -> "Tracker":
        if not self.enabled:
            return self
        import mlflow

        os.makedirs(ROOT / "mlruns" / "artifacts", exist_ok=True)
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
        if mlflow.get_experiment_by_name(self.experiment) is None:
            mlflow.create_experiment(self.experiment, artifact_location=str(ROOT / "mlruns" / "artifacts" / self.experiment))
        mlflow.set_experiment(self.experiment)
        self.run = mlflow.start_run(run_name=self.run_name, nested=self.parent is not None)
        mlflow.set_tags({**env_tags(), **self.tags})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled:
            import mlflow

            mlflow.end_run("FAILED" if exc_type else "FINISHED")

    @property
    def run_id(self) -> str | None:
        return self.run.info.run_id if self.run else None

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_params({k: (v if isinstance(v, (int, float, str, bool)) else json.dumps(v, default=str)) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_dict(self, obj: Any, name: str) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_dict(obj, name)

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_artifact(str(path), subdir)

    def log_figure(self, fig, name: str) -> None:
        if self.enabled:
            import mlflow

            mlflow.log_figure(fig, name)


@contextlib.contextmanager
def tracker(*args, **kwargs) -> Iterator[Tracker]:
    with Tracker(*args, **kwargs) as t:
        yield t
