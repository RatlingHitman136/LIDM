"""Classification / calibration metrics over a probability matrix (n_examples x n_classes)."""
from __future__ import annotations

import numpy as np


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins (same binning as Laya's)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        if (y == c).any() or (pred == c).any():
            f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def summarize(probs: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    """Accuracy, macro-F1, Brier, NLL, ECE, mean confidence.

    `probs` rows need not be exactly normalised (shortlist/tournament baselines put zero mass on
    labels they eliminated); they are renormalised and floored at `eps` for NLL.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)
    n, k = probs.shape
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y).astype(np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1(y, pred, k),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(n), y], eps, 1.0)).mean()),
        "ece": ece(conf, correct),
        "mean_confidence": float(conf.mean()),
        "n": float(n),
    }


def bootstrap_ci(correct: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = correct[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
