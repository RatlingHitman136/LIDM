import numpy as np

from cls.metrics import summarize
from cls.tracking import Tracker


def test_summarize_perfect_and_uniform():
    y = np.array([0, 1, 2])
    m = summarize(np.eye(3), y)
    assert m["accuracy"] == 1.0 and m["brier"] == 0.0 and m["ece"] < 1e-9
    u = summarize(np.full((3, 3), 1 / 3), y)
    assert abs(u["nll"] - np.log(3)) < 1e-6


def test_tracker_roundtrip(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with Tracker("t", "r", tags={"k": "v"}) as tr:
        tr.log_params({"a": 1})
        tr.log_metrics({"acc": 0.5})
    df = mlflow.search_runs(experiment_names=["t"])
    assert df.loc[0, "metrics.acc"] == 0.5 and df.loc[0, "params.a"] == "1"
