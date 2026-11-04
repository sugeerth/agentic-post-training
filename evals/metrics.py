"""Statistics for small, noisy evals. Stdlib only.

Eval sets for agent behavior are small (tens to hundreds of items), so every
headline number ships with a bootstrap CI, and comparisons are *paired* on
the same items: that removes per-item difficulty from the variance.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(
    xs: Sequence[float], *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(mean([xs[rng.randrange(k)] for _ in range(k)]) for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def paired_delta(
    baseline: Sequence[float], candidate: Sequence[float], **kw: float
) -> tuple[float, float, float]:
    """(mean delta, ci_low, ci_high) of candidate - baseline over the same items."""
    if len(baseline) != len(candidate):
        raise ValueError("paired_delta needs aligned sequences of equal length")
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    lo, hi = bootstrap_ci(diffs, **kw)  # type: ignore[arg-type]
    return (mean(diffs), lo, hi)


def brier(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of stated confidence. 0 is perfect; 0.25 is a coin flip."""
    return mean([(p - float(y)) ** 2 for p, y in zip(confidence, correct, strict=True)])


def ece(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over bins."""
    n = len(confidence)
    if n == 0:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in zip(confidence, correct, strict=True):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    return sum(
        len(b) / n * abs(mean([float(y) for _, y in b]) - mean([p for p, _ in b]))
        for b in bins
        if b
    )
