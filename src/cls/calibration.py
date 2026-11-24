"""Stage 3b: one temperature per question type and option-count bucket (docs/DESIGN.md), fitted by NLL on held-out
predictions; temperatures change confidence, never the top option. Buckets with fewer than MIN_ITEMS fall back to the
question type's temperature, and a type without data to 1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

BUCKETS = ((2, 2), (3, 4), (5, 8), (9, 16), (17, 10**6))
MIN_ITEMS = 50


def bucket(n: int) -> str:
    return next(f"{lo}-{hi}" if hi < 10**6 else f"{lo}+" for lo, hi in BUCKETS if lo <= n <= hi)


def _fit(pairs) -> float:
    """pairs: [(log-probabilities, gold)] -> the temperature minimising NLL."""
    nll = lambda t: -np.mean([(lp / t)[g] - np.log(np.exp(lp / t - (lp / t).max()).sum()) - (lp / t).max() for lp, g in pairs])
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def fit(preds: list[tuple[str, np.ndarray, int]]) -> dict:
    """preds: [(question type, probabilities, gold)] -> {"type": {...}, "type|bucket": {...}} temperatures."""
    groups: dict[str, list] = {}
    for kind, p, g in preds:
        lp = np.log(np.clip(np.asarray(p, dtype=float), 1e-9, 1.0))
        groups.setdefault(kind, []).append((lp, g))
        groups.setdefault(f"{kind}|{bucket(len(p))}", []).append((lp, g))
    return {k: {"t": _fit(v), "n": len(v)} for k, v in groups.items() if len(v) >= (1 if "|" not in k else MIN_ITEMS)}


def apply(p, kind: str, temps: dict) -> np.ndarray:
    """Rescale one question's probabilities with its bucket's temperature (else its type's, else none)."""
    p = np.asarray(p, dtype=float)
    t = (temps.get(f"{kind}|{bucket(len(p))}") or temps.get(kind) or {"t": 1.0})["t"]
    z = np.log(np.clip(p, 1e-9, 1.0)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


def load(path) -> dict:
    return json.loads(Path(path).read_text())
