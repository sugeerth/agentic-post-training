"""Which policy produced this rollout, and does the objective still hold.

GRPO and PPO are on-policy: the advantage of a sampled action is only
meaningful relative to the policy that sampled it. The moment a trainer takes
a step, every rollout still in flight was produced by a policy that no longer
exists, and the gradient computed from it is answering a question about the
wrong distribution.

Nothing crashes when this happens. The loss still falls. The reward curve
still rises. The run is simply optimizing something other than what the
objective says, and the only symptom is that it works less well than it
should — which is indistinguishable from a bad hyperparameter.

The usual defences are to pause sampling during the update, or to accept the
drift and correct for it with an importance ratio. Both are legitimate. What
is not legitimate is not knowing which one you are doing, so this module
makes the version a property of the sample rather than of the wall clock:

* every `PolicyWeights` snapshot carries the trainer step that produced it;
* every completion is stamped with the version that generated it;
* a batch assembled from mixed versions is a stated choice, made by passing
  `max_staleness`, and never a silent default.

`StalenessError` is raised rather than warned because a warning in a training
loop is a line of log nobody reads at 3am, and the failure it describes is
one that produces a plausible-looking curve either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["StalenessError", "StalenessReport", "check_staleness"]


class StalenessError(RuntimeError):
    """A batch mixes policy versions further apart than the caller allowed."""


@dataclass(frozen=True)
class StalenessReport:
    """How far the samples in a batch lag the policy about to be updated."""

    trainer_version: int
    versions: tuple[int, ...]
    #: Per-sample lag, in trainer steps.
    lags: tuple[int, ...]

    @property
    def max_lag(self) -> int:
        return max(self.lags) if self.lags else 0

    @property
    def mean_lag(self) -> float:
        return sum(self.lags) / len(self.lags) if self.lags else 0.0

    @property
    def on_policy(self) -> bool:
        """True when every sample came from the weights about to be updated."""
        return self.max_lag == 0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "trainer_version": self.trainer_version,
            "max_lag": self.max_lag,
            "mean_lag": round(self.mean_lag, 3),
            "on_policy": self.on_policy,
            "distinct_versions": len(set(self.versions)),
            "samples": len(self.versions),
        }


def check_staleness(
    versions: Sequence[int], trainer_version: int, *, max_staleness: int = 0
) -> StalenessReport:
    """Measure the lag in a batch, and refuse it if it exceeds what was allowed.

    `max_staleness=0` — the default — means strictly on-policy: every sample
    must come from the current weights. Raising it is how a caller says "I am
    running asynchronously and I accept the drift", which is a real and common
    choice. It just has to be a choice.
    """
    if not versions:
        raise ValueError("no samples to check: an empty batch has no policy")
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be >= 0, got {max_staleness}")

    lags = tuple(trainer_version - v for v in versions)
    if any(lag < 0 for lag in lags):
        ahead = max(-lag for lag in lags if lag < 0)
        raise StalenessError(
            f"a sample claims policy version {trainer_version + ahead}, ahead of "
            f"the trainer at {trainer_version}. A sampler running ahead of its "
            "trainer means the version stamps are not coming from one clock."
        )

    report = StalenessReport(trainer_version, tuple(versions), lags)
    if report.max_lag > max_staleness:
        raise StalenessError(
            f"batch lags the trainer by {report.max_lag} step(s), over the "
            f"{max_staleness} allowed. These rollouts came from a policy that "
            "no longer exists, so the on-policy objective does not hold for "
            "them. Either re-sample against the current weights, or pass "
            f"max_staleness={report.max_lag} to say the drift is intended and "
            "correct for it."
        )
    return report
