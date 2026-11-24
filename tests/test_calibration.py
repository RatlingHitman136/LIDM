"""Stage-3b temperatures (cls.calibration)."""
import numpy as np

from cls import calibration as c


def test_fit_corrects_overconfidence_and_keeps_the_argmax():
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(400):  # 3-option predictions at 0.95 confidence that are right only ~60% of the time
        g = int(rng.integers(3))
        k = g if rng.random() < 0.6 else (g + 1) % 3
        p = np.full(3, 0.025)
        p[k] = 0.95
        preds.append(("choice", p, g))
    temps = c.fit(preds)
    assert temps["choice"]["t"] > 1 and "choice|3-4" in temps
    assert abs(c.apply(np.array([0.025, 0.95, 0.025]), "choice", temps).max() - 0.6) < 0.05
    assert c.apply(np.array([0.2, 0.5, 0.3]), "choice", temps).argmax() == 1


def test_buckets_and_fallbacks():
    assert [c.bucket(n) for n in (2, 3, 4, 5, 16, 17, 150)] == ["2-2", "3-4", "3-4", "5-8", "9-16", "17+", "17+"]
    temps = {"noul": {"t": 2.0, "n": 10}}  # no bucket entry (too few items) -> the type's temperature
    assert np.allclose(c.apply(np.array([0.2, 0.8]), "noul", temps), [1 / 3, 2 / 3], atol=0.01)  # T=2: sqrt, renormalised
    assert np.allclose(c.apply(np.array([0.2, 0.8]), "score", temps), [0.2, 0.8])  # unknown type: unchanged
