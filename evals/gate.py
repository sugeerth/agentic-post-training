"""Readiness gate: is a candidate ready for inclusion in a major run?

Three rules, applied per environment on paired score deltas:

  BLOCK  any env regresses significantly: delta < -max_regression AND ci_high < 0
  HOLD   a target env fails to improve:   delta < min_gain OR ci_low <= 0
  SHIP   otherwise

Deliberately simple. A gate people can recite is a gate people trust.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evals.metrics import paired_delta


@dataclass(frozen=True, slots=True)
class EnvDelta:
    env: str
    delta: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: str  # "SHIP" | "HOLD" | "BLOCK"
    reasons: tuple[str, ...]
    deltas: tuple[EnvDelta, ...]


def decide(
    baseline: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    *,
    targets: Sequence[str],
    min_gain: float,
    max_regression: float,
) -> Decision:
    deltas = tuple(EnvDelta(env, *paired_delta(baseline[env], candidate[env])) for env in baseline)
    blocks = [
        f"{d.env} regressed {d.delta:+.3f} (CI [{d.ci_low:+.3f}, {d.ci_high:+.3f}])"
        for d in deltas
        if d.delta < -max_regression and d.ci_high < 0
    ]
    holds = [
        f"target {d.env} did not clear +{min_gain:.3f} with CI > 0 "
        f"(got {d.delta:+.3f}, CI [{d.ci_low:+.3f}, {d.ci_high:+.3f}])"
        for d in deltas
        if d.env in targets and (d.delta < min_gain or d.ci_low <= 0)
    ]
    if blocks:
        return Decision("BLOCK", tuple(blocks + holds), deltas)
    if holds:
        return Decision("HOLD", tuple(holds), deltas)
    return Decision("SHIP", ("all targets improved; no significant regressions",), deltas)
